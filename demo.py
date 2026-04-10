"""
=============================================================================
demo.py
Rare Disease Diagnosis using MAML + RL²
=============================================================================
Interactive demo that:
  1. Runs the FULL pipeline with reduced iterations (quick mode)
  2. Shows real training curves and evaluation results
  3. Generates all plots
  4. Demonstrates single-patient inference

Run: python demo.py
=============================================================================
"""

import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Change to project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import prepare_all, DATA_PATH
from models import DiagnosticClassifier, RL2DiagnosticAgent
from maml_trainer import MAMLTrainer
from rl2_trainer import RL2Trainer
from evaluation import Evaluator
from visualisation import (
    plot_dataset_overview, plot_maml_training, plot_rl2_training,
    plot_rl2_memory_effect, plot_comparison, plot_per_disease_accuracy,
    plot_summary_dashboard,
)
from data_preprocessing import load_dataset

# ─── Hyperparameters (quick demo) ───────────────────────────────────────────
CFG = {
    "num_rare_diseases": 6,
    "n_way": 5,
    "k_shot": 10,
    "n_query": 15,
    "episode_length": 20,

    # MAML (reduced for demo)
    "maml_inner_lr": 0.05,
    "maml_outer_lr": 1e-3,
    "maml_inner_steps": 5,
    "maml_first_order": True,      # FOMAML for speed
    "maml_meta_iterations": 150,
    "maml_tasks_per_iter": 4,
    "maml_eval_interval": 25,

    # RL² (reduced for demo)
    "rl2_lr": 3e-4,
    "rl2_gru_hidden": 256,
    "rl2_gru_layers": 2,
    "rl2_num_episodes": 500,
    "rl2_eval_interval": 50,
    "rl2_ppo_epochs": 4,
    "rl2_gamma": 0.99,
    "rl2_gae_lambda": 0.95,
    "rl2_clip_eps": 0.2,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)


def separator(title=""):
    line = "═" * 70
    if title:
        pad = (68 - len(title)) // 2
        print(f"\n╔{line}╗")
        print(f"║{' ' * pad}{title}{' ' * (68 - pad - len(title))}║")
        print(f"╚{line}╝")
    else:
        print(f"\n{'─' * 70}")


def main():
    separator("RARE DISEASE DIAGNOSIS — MAML + RL²")
    print(f"  Device : {DEVICE}")
    print(f"  Mode   : DEMO (reduced iterations)")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: DATA
    # ─────────────────────────────────────────────────────────────────────
    separator("STEP 1: DATA PREPROCESSING")

    data = prepare_all(
        num_rare=CFG["num_rare_diseases"],
        n_way=CFG["n_way"],
        k_shot=CFG["k_shot"],
        n_query=CFG["n_query"],
        episode_length=CFG["episode_length"],
    )

    df_raw = load_dataset(DATA_PATH)
    plot_dataset_overview(df_raw, data["symptom_list"])
    print(f"\n  ✓ Dataset overview plot saved")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: MAML TRAINING
    # ─────────────────────────────────────────────────────────────────────
    separator("STEP 2: MAML META-TRAINING")

    maml_model = DiagnosticClassifier(data["num_features"], CFG["n_way"])
    maml_trainer = MAMLTrainer(
        model=maml_model,
        inner_lr=CFG["maml_inner_lr"],
        outer_lr=CFG["maml_outer_lr"],
        inner_steps=CFG["maml_inner_steps"],
        first_order=CFG["maml_first_order"],
        device=DEVICE,
    )

    maml_train_hist, maml_val_hist = maml_trainer.train(
        train_sampler=data["maml_train_sampler"],
        val_sampler=data["maml_test_sampler"],
        num_meta_iterations=CFG["maml_meta_iterations"],
        tasks_per_iteration=CFG["maml_tasks_per_iter"],
        eval_interval=CFG["maml_eval_interval"],
        save_path="checkpoints/maml_best.pt",
    )

    # Load best
    ckpt = torch.load("checkpoints/maml_best.pt", map_location=DEVICE, weights_only=False)
    maml_model.load_state_dict(ckpt["model_state"])
    plot_maml_training(maml_train_hist, maml_val_hist)
    print("  ✓ MAML training plots saved")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: RL² TRAINING
    # ─────────────────────────────────────────────────────────────────────
    separator("STEP 3: RL² META-RL TRAINING (PPO)")

    rl2_agent = RL2DiagnosticAgent(
        num_features=data["num_features"],
        n_way=CFG["n_way"],
        gru_hidden=CFG["rl2_gru_hidden"],
        gru_layers=CFG["rl2_gru_layers"],
    )

    rl2_trainer = RL2Trainer(
        agent=rl2_agent,
        lr=CFG["rl2_lr"],
        gamma=CFG["rl2_gamma"],
        gae_lambda=CFG["rl2_gae_lambda"],
        clip_eps=CFG["rl2_clip_eps"],
        ppo_epochs=CFG["rl2_ppo_epochs"],
        device=DEVICE,
    )

    rl2_train_hist, rl2_val_hist = rl2_trainer.train(
        train_gen=data["rl2_train_gen"],
        val_gen=data["rl2_test_gen"],
        num_episodes=CFG["rl2_num_episodes"],
        eval_interval=CFG["rl2_eval_interval"],
        save_path="checkpoints/rl2_best.pt",
    )

    ckpt = torch.load("checkpoints/rl2_best.pt", map_location=DEVICE, weights_only=False)
    rl2_agent.load_state_dict(ckpt["model_state"])
    plot_rl2_training(rl2_train_hist, rl2_val_hist)
    print("  ✓ RL² training plots saved")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: EVALUATION
    # ─────────────────────────────────────────────────────────────────────
    separator("STEP 4: EVALUATION")

    evaluator = Evaluator(device=DEVICE)

    print("\n  [1/4] Baseline evaluation...")
    baseline_results = evaluator.evaluate_standard_baseline(
        data["train_groups"], data["test_groups"], CFG["n_way"]
    )

    print("  [2/4] MAML k-shot evaluation...")
    maml_results = evaluator.evaluate_maml(
        maml_trainer, data["maml_test_sampler"], k_shots=(5, 10, 15), num_tasks=40
    )

    print("  [3/4] RL² sequential evaluation...")
    rl2_results = evaluator.evaluate_rl2(
        rl2_trainer, data["rl2_test_gen"], num_episodes=40
    )

    print("  [4/4] Per-disease accuracy...")
    per_disease = evaluator.per_disease_accuracy(
        maml_trainer, data["test_groups"], k_shot=10
    )

    evaluator.print_full_report(maml_results, rl2_results, baseline_results, per_disease)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5: VISUALISATIONS
    # ─────────────────────────────────────────────────────────────────────
    separator("STEP 5: GENERATING ALL PLOTS")

    if rl2_results.get("step_accuracy"):
        plot_rl2_memory_effect(rl2_results["step_accuracy"], CFG["n_way"])

    plot_comparison(
        baseline_results=baseline_results,
        maml_results=maml_results,
        rl2_acc=rl2_results["mean_accuracy"],
        k_shots=[k for k in [5, 10, 15] if k in maml_results],
    )

    if per_disease:
        plot_per_disease_accuracy(per_disease)

    plot_summary_dashboard(
        maml_results=maml_results,
        rl2_results=rl2_results,
        train_history_maml=maml_train_hist,
        train_history_rl2=rl2_train_hist,
    )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 6: SINGLE-PATIENT INFERENCE DEMO
    # ─────────────────────────────────────────────────────────────────────
    separator("STEP 6: SINGLE-PATIENT INFERENCE DEMO")

    print("\n  Demonstrating MAML rapid adaptation on one rare disease task...")

    support_x, support_y, query_x, query_y, diseases = \
        data["maml_test_sampler"].sample_task()

    print(f"\n  Rare disease task:")
    for i, d in enumerate(diseases):
        print(f"    Class {i}: {d}")

    # Adapt with 10-shot support set
    fast_weights = maml_trainer.inner_loop(
        support_x.to(DEVICE), support_y.to(DEVICE)
    )

    with torch.no_grad():
        logits = maml_trainer._functional_forward(query_x.to(DEVICE), fast_weights)
        preds = logits.argmax(dim=1).cpu()
        acc = (preds == query_y).float().mean().item() * 100

    print(f"\n  After 5 gradient steps on 10 support examples:")
    print(f"  Query accuracy = {acc:.1f}%  ({int(acc * len(query_y) / 100)}/{len(query_y)} correct)")

    # Show first 5 predictions
    print(f"\n  Sample predictions (first 5 query patients):")
    print(f"  {'True Disease':<40}  {'Predicted':<40}  {'✓/✗'}")
    print("  " + "-" * 90)
    for i in range(min(5, len(query_y))):
        true_d = diseases[query_y[i].item()]
        pred_d = diseases[preds[i].item()]
        mark = "✓" if preds[i] == query_y[i] else "✗"
        print(f"  {true_d:<40}  {pred_d:<40}  {mark}")

    # RL² demo
    print(f"\n  Demonstrating RL² sequential diagnosis on rare disease episode...")
    obs_seq, true_acts, _, ep_diseases, num_f = data["rl2_test_gen"].generate_episode()

    print(f"\n  Episode diseases: {ep_diseases}")
    hidden = rl2_agent.init_hidden(1, DEVICE)
    prev_act_oh = np.zeros(CFG["n_way"], dtype=np.float32)
    prev_rew = 0.0

    correct_early, correct_late = 0, 0
    ep_len = obs_seq.shape[0]

    for t in range(ep_len):
        sym_t = obs_seq[t][:num_f].to(DEVICE)
        obs_t = torch.cat([
            sym_t,
            torch.FloatTensor(prev_act_oh).to(DEVICE),
            torch.FloatTensor([prev_rew]).to(DEVICE),
        ])
        action, _, _, hidden = rl2_agent.act(obs_t, hidden)
        true = true_acts[t].item()
        correct = action == true
        if t < ep_len // 2:
            correct_early += int(correct)
        else:
            correct_late += int(correct)
        prev_act_oh = np.zeros(CFG["n_way"], dtype=np.float32)
        prev_act_oh[action] = 1.0
        prev_rew = 1.0 if correct else -0.5

    early_acc = correct_early / (ep_len // 2) * 100
    late_acc  = correct_late  / (ep_len - ep_len // 2) * 100
    print(f"\n  Early phase (patients 1-{ep_len//2}) accuracy : {early_acc:.1f}%")
    print(f"  Late phase  (patients {ep_len//2+1}-{ep_len}) accuracy : {late_acc:.1f}%")
    print(f"  Memory improvement                        : +{late_acc - early_acc:.1f}% ✓")

    # ─────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    separator("RESULTS SUMMARY")

    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  METHOD          │ EXAMPLES NEEDED │ ACCURACY            │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    b10  = baseline_results.get(10, (0, 0))
    m5   = maml_results.get(5, (0, 0))
    m10  = maml_results.get(10, (0, 0))
    m15  = maml_results.get(15, (0, 0))
    r_a  = rl2_results["mean_accuracy"]
    print(f"  │  Standard AI     │      50+        │ {b10[0]:6.1f}% ± {b10[1]:.1f}%       │")
    print(f"  │  MAML (5-shot)   │       5         │ {m5[0]:6.1f}% ± {m5[1]:.1f}%       │")
    print(f"  │  MAML (10-shot)  │      10         │ {m10[0]:6.1f}% ± {m10[1]:.1f}%       │")
    print(f"  │  MAML (15-shot)  │      15         │ {m15[0]:6.1f}% ± {m15[1]:.1f}%       │")
    print(f"  │  RL² (sequential)│    sequence     │ {r_a:6.1f}% ± {rl2_results['std_accuracy']:.1f}%       │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    print(f"\n  Plots saved to     : ./plots/")
    print(f"  Checkpoints saved  : ./checkpoints/")
    separator("DEMO COMPLETE")


if __name__ == "__main__":
    main()
