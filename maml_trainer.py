"""
=============================================================================
maml_trainer.py
Rare Disease Diagnosis using MAML + RL²
=============================================================================
Implements the full MAML (Model-Agnostic Meta-Learning) training loop:

  OUTER LOOP: updates global model parameters θ across a batch of disease tasks
  INNER LOOP: for each task, adapts θ → θ' using the support set

Algorithm (MAML, Finn et al. 2017):
  For each meta-iteration:
    Sample batch of tasks {T_1, ..., T_B}
    For each task T_i:
      Compute adapted params: θ'_i = θ - α ∇_θ L_support(f_θ)
    Meta-update: θ ← θ - β ∇_θ Σ_i L_query(f_θ'_i)

Key design choices:
  - first_order=True option for First-Order MAML (FOMAML) for speed
  - inner_lr is task-specific and learned (Meta-SGD variant)
  - Support loss: NLL on support set | Query loss: NLL on query set
  - Accuracy tracked on query set for monitoring
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import defaultdict
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# MAML TRAINER
# ─────────────────────────────────────────────────────────────────────────────
class MAMLTrainer:
    """
    Full MAML training and evaluation loop.

    Parameters
    ----------
    model          : DiagnosticClassifier — the base learner
    inner_lr       : float  — inner loop (task-specific) learning rate (α)
    outer_lr       : float  — outer loop (meta) learning rate (β)
    inner_steps    : int    — number of gradient steps in the inner loop
    first_order    : bool   — use FOMAML (faster, slightly less accurate)
    device         : torch.device
    """

    def __init__(self, model, inner_lr=0.05, outer_lr=1e-3,
                 inner_steps=5, first_order=False, device=None):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Meta-optimizer (outer loop)
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.outer_lr,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.meta_optimizer, T_max=200, eta_min=1e-5
        )

        # History
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)

    # ─────────────────────────────────────────────
    # INNER LOOP (Task Adaptation)
    # ─────────────────────────────────────────────
    def inner_loop(self, support_x, support_y):
        """
        Performs k gradient steps on the support set.
        Returns the adapted parameters θ' (as an OrderedDict).

        This is the "fast adaptation" step of MAML.

        support_x : Tensor (n_way*k_shot, num_features)
        support_y : Tensor (n_way*k_shot,)
        """
        # Clone model parameters — we adapt the clone, not the original
        fast_weights = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }

        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)

        for step in range(self.inner_steps):
            # Forward pass using current fast_weights
            # We call functional_forward to use the fast weights
            logits = self._functional_forward(support_x, fast_weights)
            loss = F.nll_loss(logits, support_y)

            # Compute gradients w.r.t. fast_weights
            grads = torch.autograd.grad(
                loss,
                fast_weights.values(),
                create_graph=not self.first_order,  # 2nd order grads for full MAML
                allow_unused=True,
            )

            # Update fast_weights: θ' ← θ' - α * ∇L
            fast_weights = {
                name: param - self.inner_lr * (grad if grad is not None else torch.zeros_like(param))
                for (name, param), grad in zip(fast_weights.items(), grads)
            }

        return fast_weights

    # ─────────────────────────────────────────────
    # FUNCTIONAL FORWARD
    # ─────────────────────────────────────────────
    def _functional_forward(self, x, params):
        """
        Runs forward pass using custom parameter dict.
        Mimics model.forward() but with provided params.
        """
        x_current = x

        # Walk through the network layers
        for name, module in self.model.network.named_modules():
            if name == "":
                continue

            full_key = f"network.{name}"

            if isinstance(module, nn.Linear):
                w = params.get(f"{full_key}.weight")
                b = params.get(f"{full_key}.bias")
                if w is not None:
                    x_current = F.linear(x_current, w, b)

            elif isinstance(module, nn.BatchNorm1d):
                w = params.get(f"{full_key}.weight")
                b = params.get(f"{full_key}.bias")
                if w is not None:
                    x_current = F.batch_norm(
                        x_current,
                        running_mean=module.running_mean.detach(),
                        running_var=module.running_var.detach(),
                        weight=w, bias=b,
                        training=True,
                        eps=module.eps,
                        momentum=module.momentum,
                    )

            elif isinstance(module, nn.ReLU):
                x_current = F.relu(x_current, inplace=False)

            elif isinstance(module, nn.Dropout):
                x_current = F.dropout(x_current, p=module.p,
                                      training=self.model.training)

        return F.log_softmax(x_current, dim=-1)

    # ─────────────────────────────────────────────
    # META-TRAINING ITERATION
    # ─────────────────────────────────────────────
    def meta_train_step(self, task_batch):
        """
        One outer-loop meta-update step.

        task_batch : list of (support_x, support_y, query_x, query_y, diseases)

        Returns:
          meta_loss   : float — mean query loss across tasks
          meta_acc    : float — mean query accuracy across tasks
        """
        self.model.train()
        self.meta_optimizer.zero_grad()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for support_x, support_y, query_x, query_y, _ in task_batch:
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)

            # ── INNER LOOP: adapt to this task ──────────────────────────
            fast_weights = self.inner_loop(support_x, support_y)

            # ── QUERY EVALUATION with adapted weights ───────────────────
            query_logits = self._functional_forward(query_x, fast_weights)
            task_loss = F.nll_loss(query_logits, query_y)

            total_loss += task_loss

            # Accuracy
            preds = query_logits.argmax(dim=1)
            total_correct += (preds == query_y).sum().item()
            total_samples += len(query_y)

        # Average loss across tasks
        meta_loss = total_loss / len(task_batch)

        # ── OUTER LOOP: meta-gradient update ────────────────────────────
        meta_loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.meta_optimizer.step()

        meta_acc = total_correct / total_samples
        return meta_loss.item(), meta_acc

    # ─────────────────────────────────────────────
    # META-EVALUATION (on rare/held-out diseases)
    # ─────────────────────────────────────────────
    def meta_evaluate(self, task_sampler, num_eval_tasks=50):
        """
        Evaluates MAML on held-out rare disease tasks.
        For each task: adapts on support set → evaluates on query set.

        Returns dict with accuracy at 1-shot, 5-shot, 10-shot.
        """
        self.model.eval()
        results = defaultdict(list)

        for _ in range(num_eval_tasks):
            support_x, support_y, query_x, query_y, diseases = \
                task_sampler.sample_task()

            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)

            # Test different k-shot regimes
            for k_shot in [5, 10, 15]:
                # Subsample support set to k_shot per class
                n_way = len(diseases)
                sub_sx, sub_sy = [], []
                for c in range(n_way):
                    mask = (support_y == c).nonzero(as_tuple=True)[0]
                    k_actual = min(k_shot, len(mask))
                    chosen = mask[:k_actual]
                    sub_sx.append(support_x[chosen])
                    sub_sy.append(support_y[chosen])

                sub_sx = torch.cat(sub_sx).to(self.device)
                sub_sy = torch.cat(sub_sy).to(self.device)

                # Adapt
                fast_weights = self.inner_loop(sub_sx, sub_sy)

                # Evaluate
                with torch.no_grad():
                    logits = self._functional_forward(query_x, fast_weights)
                    preds = logits.argmax(dim=1)
                    acc = (preds == query_y).float().mean().item()

                results[f"acc_{k_shot}shot"].append(acc)

        summary = {
            k: (np.mean(v) * 100, np.std(v) * 100)
            for k, v in results.items()
        }
        return summary

    # ─────────────────────────────────────────────
    # FULL TRAINING LOOP
    # ─────────────────────────────────────────────
    def train(self, train_sampler, val_sampler,
              num_meta_iterations=300,
              tasks_per_iteration=8,
              eval_interval=25,
              save_path="checkpoints/maml_best.pt"):
        """
        Full MAML meta-training loop.

        Parameters
        ----------
        train_sampler        : MAMLTaskSampler for meta-training diseases
        val_sampler          : MAMLTaskSampler for rare disease evaluation
        num_meta_iterations  : total number of outer-loop updates
        tasks_per_iteration  : tasks sampled per outer-loop step
        eval_interval        : evaluate every N iterations
        save_path            : where to save the best model
        """
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        best_val_acc = 0.0

        print("\n" + "=" * 70)
        print("  MAML META-TRAINING")
        print(f"  Iterations: {num_meta_iterations} | Tasks/iter: {tasks_per_iteration}")
        print(f"  Inner LR: {self.inner_lr} | Outer LR: {self.outer_lr}")
        print(f"  Inner Steps: {self.inner_steps} | First-Order: {self.first_order}")
        print("=" * 70)

        pbar = tqdm(range(1, num_meta_iterations + 1), desc="MAML Training")

        for iteration in pbar:
            # Sample task batch
            task_batch = train_sampler.sample_batch(tasks_per_iteration)

            # Meta-train step
            loss, acc = self.meta_train_step(task_batch)

            self.train_history["loss"].append(loss)
            self.train_history["acc"].append(acc)

            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "acc": f"{acc*100:.1f}%",
            })

            # Periodic evaluation
            if iteration % eval_interval == 0:
                val_results = self.meta_evaluate(val_sampler, num_eval_tasks=30)
                val_acc_10 = val_results.get("acc_10shot", (0, 0))[0]

                self.val_history["iteration"].append(iteration)
                for k, (mean, std) in val_results.items():
                    self.val_history[k].append(mean)

                print(f"\n[Iter {iteration:4d}] "
                      f"Train Loss: {loss:.4f} | Train Acc: {acc*100:.1f}% | "
                      f"Val 5-shot: {val_results.get('acc_5shot',(0,0))[0]:.1f}% | "
                      f"Val 10-shot: {val_acc_10:.1f}% | "
                      f"Val 15-shot: {val_results.get('acc_15shot',(0,0))[0]:.1f}%")

                # Save best model
                if val_acc_10 > best_val_acc:
                    best_val_acc = val_acc_10
                    torch.save({
                        "iteration": iteration,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.meta_optimizer.state_dict(),
                        "val_acc": best_val_acc,
                        "config": {
                            "inner_lr": self.inner_lr,
                            "outer_lr": self.outer_lr,
                            "inner_steps": self.inner_steps,
                        }
                    }, save_path)
                    print(f"  ★ New best model saved (10-shot acc: {best_val_acc:.1f}%)")

            self.scheduler.step()

        print(f"\n[MAML] Training complete. Best 10-shot val accuracy: {best_val_acc:.2f}%")
        return self.train_history, self.val_history
