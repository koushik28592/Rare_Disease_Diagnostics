"""
=============================================================================
evaluation.py
Rare Disease Diagnosis using MAML + RL²
=============================================================================
Comprehensive evaluation of:
  1. MAML adaptation performance (k-shot accuracy on rare diseases)
  2. RL² sequential diagnostic accuracy (per-step and cumulative)
  3. Hybrid MAML+RL² combined performance
  4. Comparison table: Standard AI vs MAML-only vs RL²-only vs Hybrid

Also includes:
  - Confusion matrix generation
  - Per-disease accuracy breakdown
  - Learning curve analysis
=============================================================================
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


class Evaluator:
    """
    Comprehensive evaluation suite for all three model configurations.
    """

    def __init__(self, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ─────────────────────────────────────────────
    # 1. STANDARD BASELINE EVALUATION
    # ─────────────────────────────────────────────
    def evaluate_standard_baseline(self, train_groups, test_groups, n_way=5):
        """
        Trains a standard neural net from scratch on k training examples,
        tests on query set. No meta-learning. Uses same architecture.
        """
        from models import DiagnosticClassifier

        k_shots = [5, 10, 15, 50]
        results = {}

        test_diseases = list(test_groups.keys())

        for k_shot in k_shots:
            accs = []

            for _ in range(20):  # 20 random tasks
                import random
                selected = random.sample(test_diseases, min(n_way, len(test_diseases)))

                # Build train and test sets
                train_x, train_y = [], []
                test_x, test_y = [], []

                for local_label, disease in enumerate(selected):
                    pool = test_groups[disease]
                    if len(pool) < k_shot + 10:
                        idx = np.random.choice(len(pool), k_shot + 10, replace=True)
                    else:
                        idx = np.random.choice(len(pool), k_shot + 10, replace=False)

                    train_x.append(pool[idx[:k_shot]])
                    train_y.extend([local_label] * k_shot)
                    test_x.append(pool[idx[k_shot:]])
                    test_y.extend([local_label] * 10)

                train_x = torch.FloatTensor(np.vstack(train_x)).to(self.device)
                train_y = torch.LongTensor(train_y).to(self.device)
                test_x  = torch.FloatTensor(np.vstack(test_x)).to(self.device)
                test_y  = torch.LongTensor(test_y).to(self.device)

                num_features = train_x.shape[1]
                model = DiagnosticClassifier(num_features, len(selected)).to(self.device)
                opt = torch.optim.Adam(model.parameters(), lr=0.01)

                # Train for 100 steps
                model.train()
                for _ in range(100):
                    logits = model(train_x)
                    loss = F.nll_loss(logits, train_y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                # Test
                model.eval()
                with torch.no_grad():
                    logits = model(test_x)
                    preds = logits.argmax(dim=1)
                    acc = (preds == test_y).float().mean().item()
                accs.append(acc)

            results[k_shot] = (np.mean(accs) * 100, np.std(accs) * 100)

        return results

    # ─────────────────────────────────────────────
    # 2. MAML EVALUATION (k-shot)
    # ─────────────────────────────────────────────
    def evaluate_maml(self, maml_trainer, test_sampler,
                      k_shots=(5, 10, 15), num_tasks=50):
        """
        Evaluate MAML at multiple k-shot settings on rare diseases.
        """
        from maml_trainer import MAMLTrainer

        results = {}
        maml_trainer.model.eval()

        for k in k_shots:
            accs = []

            for _ in range(num_tasks):
                support_x, support_y, query_x, query_y, diseases = \
                    test_sampler.sample_task()

                n_way = len(diseases)
                query_x = query_x.to(self.device)
                query_y = query_y.to(self.device)

                # Sub-sample to exactly k-shot
                sub_sx, sub_sy = [], []
                for c in range(n_way):
                    mask = (support_y == c).nonzero(as_tuple=True)[0]
                    k_actual = min(k, len(mask))
                    chosen = mask[:k_actual]
                    sub_sx.append(support_x[chosen])
                    sub_sy.append(support_y[chosen])

                sub_sx = torch.cat(sub_sx).to(self.device)
                sub_sy = torch.cat(sub_sy).to(self.device)

                # Adapt
                fast_weights = maml_trainer.inner_loop(sub_sx, sub_sy)

                with torch.no_grad():
                    logits = maml_trainer._functional_forward(query_x, fast_weights)
                    preds = logits.argmax(dim=1)
                    acc = (preds == query_y).float().mean().item()

                accs.append(acc)

            results[k] = (np.mean(accs) * 100, np.std(accs) * 100)

        return results

    # ─────────────────────────────────────────────
    # 3. RL² EVALUATION (sequential)
    # ─────────────────────────────────────────────
    def evaluate_rl2(self, rl2_trainer, test_gen, num_episodes=50):
        """
        Evaluate RL² agent on rare disease episodes.
        Returns per-step accuracy (shows learning within episode).
        """
        rl2_trainer.agent.eval()

        all_step_accs = defaultdict(list)
        all_episode_accs = []
        all_rewards = []

        for _ in range(num_episodes):
            obs_seq, true_actions, _, _, num_features = test_gen.generate_episode()
            episode_length = obs_seq.shape[0]
            n_way = rl2_trainer.agent.n_way

            hidden = rl2_trainer.agent.init_hidden(batch_size=1,
                                                    device=self.device)
            prev_action_oh = np.zeros(n_way, dtype=np.float32)
            prev_reward_val = 0.0

            ep_correct = 0
            ep_reward = 0.0

            for t in range(episode_length):
                sym_t = obs_seq[t][:num_features].to(self.device)
                obs_t = torch.cat([
                    sym_t,
                    torch.FloatTensor(prev_action_oh).to(self.device),
                    torch.FloatTensor([prev_reward_val]).to(self.device),
                ])

                action, _, _, hidden = rl2_trainer.agent.act(obs_t, hidden)
                true = true_actions[t].item()
                correct = int(action == true)
                reward = 1.0 if correct else -0.5

                all_step_accs[t].append(correct)
                ep_correct += correct
                ep_reward += reward

                prev_action_oh = np.zeros(n_way, dtype=np.float32)
                prev_action_oh[action] = 1.0
                prev_reward_val = reward

            all_episode_accs.append(ep_correct / episode_length)
            all_rewards.append(ep_reward)

        rl2_trainer.agent.train()

        # Per-step accuracy: shows how agent improves as it sees more patients
        step_acc = {t: np.mean(v) * 100 for t, v in all_step_accs.items()}

        return {
            "mean_accuracy": np.mean(all_episode_accs) * 100,
            "std_accuracy": np.std(all_episode_accs) * 100,
            "mean_reward": np.mean(all_rewards),
            "step_accuracy": step_acc,
        }

    # ─────────────────────────────────────────────
    # 4. COMPARISON TABLE
    # ─────────────────────────────────────────────
    def generate_comparison_table(self, baseline_results, maml_results,
                                   rl2_results, hybrid_results=None):
        """
        Builds the comparison table from the paper's expected results.
        Returns a pandas DataFrame.
        """
        rows = []

        # k-shot values to report
        k_values = sorted(set(baseline_results.keys()) & set(maml_results.keys()))

        for k in k_values:
            b_mean, b_std = baseline_results.get(k, (0, 0))
            m_mean, m_std = maml_results.get(k, (0, 0))
            r_mean = rl2_results.get("mean_accuracy", 0)
            r_std = rl2_results.get("std_accuracy", 0)
            h_mean = hybrid_results.get("mean_accuracy", 0) if hybrid_results else "N/A"

            rows.append({
                "K-Shot": k,
                "Standard AI (%)": f"{b_mean:.1f} ± {b_std:.1f}",
                "MAML Only (%)": f"{m_mean:.1f} ± {m_std:.1f}",
                "RL² Only (%)": f"{r_mean:.1f} ± {r_std:.1f}",
                "MAML + RL² (%)": f"{h_mean:.1f}" if isinstance(h_mean, float) else h_mean,
            })

        df = pd.DataFrame(rows)
        return df

    # ─────────────────────────────────────────────
    # 5. PER-DISEASE ACCURACY (MAML)
    # ─────────────────────────────────────────────
    def per_disease_accuracy(self, maml_trainer, test_groups,
                              k_shot=10, n_queries=20, inner_steps=5):
        """
        Report MAML accuracy separately for each rare disease.
        """
        results = {}
        maml_trainer.model.eval()

        for disease, X_pool in test_groups.items():
            accs = []

            for _ in range(20):
                if len(X_pool) < k_shot + n_queries:
                    idx = np.random.choice(len(X_pool), k_shot + n_queries, replace=True)
                else:
                    idx = np.random.choice(len(X_pool), k_shot + n_queries, replace=False)

                # 2-way: disease vs random noise (binary)
                # For simplicity, evaluate as 1-class confidence
                sx = torch.FloatTensor(X_pool[idx[:k_shot]]).to(self.device)
                qx = torch.FloatTensor(X_pool[idx[k_shot:]]).to(self.device)

                # Single-disease accuracy: does model correctly identify this disease?
                # We use a 2-way task: this disease vs one random other
                import random
                other_disease = random.choice(
                    [d for d in test_groups if d != disease]
                )
                other_pool = test_groups[other_disease]
                other_sx = torch.FloatTensor(
                    other_pool[np.random.choice(len(other_pool), k_shot, replace=True)]
                ).to(self.device)
                other_qx = torch.FloatTensor(
                    other_pool[np.random.choice(len(other_pool), n_queries, replace=True)]
                ).to(self.device)

                full_sx = torch.cat([sx, other_sx])
                full_sy = torch.LongTensor([0] * k_shot + [1] * k_shot).to(self.device)
                full_qx = torch.cat([qx, other_qx])
                full_qy = torch.LongTensor([0] * n_queries + [1] * n_queries).to(self.device)

                fast_weights = maml_trainer.inner_loop(full_sx, full_sy)
                with torch.no_grad():
                    logits = maml_trainer._functional_forward(full_qx, fast_weights)
                    preds = logits.argmax(dim=1)
                    acc = (preds == full_qy).float().mean().item()

                accs.append(acc)

            results[disease] = (np.mean(accs) * 100, np.std(accs) * 100)

        return results

    # ─────────────────────────────────────────────
    # 6. PRINT FULL REPORT
    # ─────────────────────────────────────────────
    def print_full_report(self, maml_results, rl2_results,
                           baseline_results, per_disease_results):
        """Prints a nicely formatted evaluation report."""
        print("\n" + "=" * 70)
        print("  EVALUATION REPORT — RARE DISEASE DIAGNOSIS WITH MAML + RL²")
        print("=" * 70)

        print("\n📊 MAML K-SHOT ACCURACY ON RARE DISEASES:")
        print(f"  {'K-Shot':<10} {'Accuracy':<25} {'95% CI'}")
        print("  " + "-" * 40)
        for k, (mean, std) in sorted(maml_results.items()):
            ci = 1.96 * std / np.sqrt(50)
            print(f"  {k:<10} {mean:.2f}% ± {std:.2f}%{'':<8} ± {ci:.2f}%")

        print("\n🧠 RL² SEQUENTIAL DIAGNOSTIC PERFORMANCE:")
        print(f"  Mean Accuracy : {rl2_results['mean_accuracy']:.2f}% ± {rl2_results['std_accuracy']:.2f}%")
        print(f"  Mean Reward   : {rl2_results['mean_reward']:.4f}")

        if rl2_results.get("step_accuracy"):
            steps = sorted(rl2_results["step_accuracy"].keys())
            early = np.mean([rl2_results["step_accuracy"][t] for t in steps[:5]])
            late  = np.mean([rl2_results["step_accuracy"][t] for t in steps[-5:]])
            print(f"  Early Steps (1-5) Acc  : {early:.1f}%")
            print(f"  Late Steps (-5) Acc    : {late:.1f}%")
            print(f"  Improvement            : +{late - early:.1f}% (GRU memory effect)")

        if baseline_results:
            print("\n📈 COMPARISON: STANDARD AI vs MAML:")
            print(f"  {'K-Shot':<8} {'Standard AI':<20} {'MAML':<20} {'Improvement'}")
            print("  " + "-" * 55)
            for k in sorted(set(baseline_results.keys()) & set(maml_results.keys())):
                b_m, b_s = baseline_results[k]
                m_m, m_s = maml_results[k]
                gain = m_m - b_m
                print(f"  {k:<8} {b_m:.1f}% ± {b_s:.1f}%{'':<8} "
                      f"{m_m:.1f}% ± {m_s:.1f}%{'':<8} +{gain:.1f}%")

        if per_disease_results:
            print("\n🔬 PER RARE DISEASE ACCURACY (MAML, 10-shot):")
            for disease, (mean, std) in sorted(per_disease_results.items(),
                                                key=lambda x: -x[1][0]):
                bar = "█" * int(mean / 5)
                print(f"  {disease:<40} {mean:5.1f}% ± {std:.1f}%  {bar}")

        print("\n" + "=" * 70)
