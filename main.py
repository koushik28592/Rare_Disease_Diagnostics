"""
=============================================================================
main.py
Rare Disease Diagnosis using MAML + RL²
=============================================================================
Master training script. Runs the complete pipeline:

  Phase 1: Data Preprocessing
  Phase 2: MAML Meta-Training  (learns HOW to diagnose)
  Phase 3: RL² Meta-RL Training (learns sequential diagnostic strategy)
  Phase 4: Evaluation & Comparison
  Phase 5: Visualisation

Usage:
  python main.py                     # full training
  python main.py --quick             # reduced iterations (for testing)
  python main.py --eval-only         # evaluation from saved checkpoints
=============================================================================
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

# ─── Project imports ────────────────────────────────────────────────────────
from data_preprocessing import prepare_all, DATA_PATH
from models import DiagnosticClassifier, RL2DiagnosticAgent, HybridMAMLRL2Agent
from maml_trainer import MAMLTrainer
from rl2_trainer import RL2Trainer
from evaluation import Evaluator
from visualisation import (
    plot_dataset_overview,
    plot_maml_training,
    plot_rl2_training,
    plot_rl2_memory_effect,
    plot_comparison,
    plot_per_disease_accuracy,
    plot_summary_dashboard,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
def get_config(quick=False):
    return {
        # Data
        "num_rare_diseases": 6,
        "n_way": 5,
        "k_shot": 10,
        "n_query": 15,
        "episode_length": 20,

        # MAML
        "maml_inner_lr": 0.05,
        "maml_outer_lr": 1e-3,
        "maml_inner_steps": 5,
        "maml_first_order": False,
        "maml_meta_iterations": 50 if quick else 300,
        "maml_tasks_per_iter": 4 if quick else 8,
        "maml_eval_interval": 10 if quick else 25,

        # RL²
        "rl2_lr": 3e-4,
        "rl2_gru_hidden": 256,
        "rl2_gru_layers": 2,
        "rl2_num_episodes": 200 if quick else 1000,
        "rl2_eval_interval": 50 if quick else 100,
        "rl2_ppo_epochs": 4,
        "rl2_gamma": 0.99,
        "rl2_gae_lambda": 0.95,
        "rl2_clip_eps": 0.2,

        # Checkpoints
        "maml_ckpt": "checkpoints/maml_best.pt",
        "rl2_ckpt":  "checkpoints/rl2_best.pt",

        # Eval
        "num_eval_tasks": 50,
        "num_eval_episodes": 50,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def phase1_data(cfg):
    print("\n" + "█" * 70)
    print("  PHASE 1: DATA PREPROCESSING")
    print("█" * 70)

    import pandas as pd
    from data_preprocessing import build_feature_matrix, load_dataset

    data = prepare_all(
    data_path="data/DiseaseAndSymptoms.csv",
    num_rare=cfg["num_rare_diseases"],
    n_way=cfg["n_way"],
    k_shot=cfg["k_shot"],
    n_query=cfg["n_query"],
    episode_length=cfg["episode_length"],
)

    # EDA plot
    df_raw = load_dataset(DATA_PATH)
    plot_dataset_overview(df_raw, data["symptom_list"])

    print(f"\n  Feature dimensions   : {data['num_features']}")
    print(f"  Meta-train diseases  : {len(data['train_diseases'])}")
    print(f"  Meta-test diseases   : {len(data['test_diseases'])}")
    print(f"  n_way                : {cfg['n_way']}")
    print(f"  k_shot               : {cfg['k_shot']}")
    print(f"  Episode length (RL²) : {cfg['episode_length']}")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: MAML META-TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def phase2_maml(cfg, data, device):
    print("\n" + "█" * 70)
    print("  PHASE 2: MAML META-TRAINING")
    print("█" * 70)

    # Build model
    model = DiagnosticClassifier(
        num_features=data["num_features"],
        n_way=cfg["n_way"],
    )

    # Total params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  MAML Base Model parameters: {n_params:,}")

    # Trainer
    trainer = MAMLTrainer(
        model=model,
        inner_lr=cfg["maml_inner_lr"],
        outer_lr=cfg["maml_outer_lr"],
        inner_steps=cfg["maml_inner_steps"],
        first_order=cfg["maml_first_order"],
        device=device,
    )

    # Train
    train_hist, val_hist = trainer.train(
        train_sampler=data["maml_train_sampler"],
        val_sampler=data["maml_test_sampler"],
        num_meta_iterations=cfg["maml_meta_iterations"],
        tasks_per_iteration=cfg["maml_tasks_per_iter"],
        eval_interval=cfg["maml_eval_interval"],
        save_path=cfg["maml_ckpt"],
    )

    # Load best checkpoint
    ckpt = torch.load(cfg["maml_ckpt"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"\n  Best MAML checkpoint: iteration {ckpt['iteration']}, "
          f"val acc: {ckpt['val_acc']:.2f}%")

    # Plots
    plot_maml_training(train_hist, val_hist)

    return trainer, train_hist, val_hist


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: RL² META-RL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def phase3_rl2(cfg, data, device):
    print("\n" + "█" * 70)
    print("  PHASE 3: RL² META-REINFORCEMENT LEARNING TRAINING")
    print("█" * 70)

    # Build RL² agent
    agent = RL2DiagnosticAgent(
        num_features=data["num_features"],
        n_way=cfg["n_way"],
        gru_hidden=cfg["rl2_gru_hidden"],
        gru_layers=cfg["rl2_gru_layers"],
    )

    n_params = sum(p.numel() for p in agent.parameters())
    obs_dim = data["num_features"] + cfg["n_way"] + 1
    print(f"\n  RL² Agent parameters : {n_params:,}")
    print(f"  Observation dim      : {obs_dim} "
          f"({data['num_features']} symptoms + {cfg['n_way']} action_oh + 1 reward)")
    print(f"  GRU hidden           : {cfg['rl2_gru_hidden']} × {cfg['rl2_gru_layers']} layers")

    # Trainer
    trainer = RL2Trainer(
        agent=agent,
        lr=cfg["rl2_lr"],
        gamma=cfg["rl2_gamma"],
        gae_lambda=cfg["rl2_gae_lambda"],
        clip_eps=cfg["rl2_clip_eps"],
        ppo_epochs=cfg["rl2_ppo_epochs"],
        device=device,
    )

    # Train
    train_hist, val_hist = trainer.train(
        train_gen=data["rl2_train_gen"],
        val_gen=data["rl2_test_gen"],
        num_episodes=cfg["rl2_num_episodes"],
        eval_interval=cfg["rl2_eval_interval"],
        save_path=cfg["rl2_ckpt"],
    )

    # Load best checkpoint
    ckpt = torch.load(cfg["rl2_ckpt"], map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["model_state"])
    print(f"\n  Best RL² checkpoint: episode {ckpt['episode']}, "
          f"val acc: {ckpt['val_acc']:.2f}%")

    # Plots
    plot_rl2_training(train_hist, val_hist)

    return trainer, train_hist, val_hist


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def phase4_evaluation(cfg, data, maml_trainer, rl2_trainer, device):
    print("\n" + "█" * 70)
    print("  PHASE 4: EVALUATION & COMPARISON")
    print("█" * 70)

    evaluator = Evaluator(device=device)

    # 1. Standard baseline
    print("\n  Evaluating standard AI baseline (from scratch)...")
    baseline_results = evaluator.evaluate_standard_baseline(
        train_groups=data["train_groups"],
        test_groups=data["test_groups"],
        n_way=cfg["n_way"],
    )

    # 2. MAML evaluation
    print("\n  Evaluating MAML on rare diseases...")
    maml_results = evaluator.evaluate_maml(
        maml_trainer=maml_trainer,
        test_sampler=data["maml_test_sampler"],
        k_shots=(5, 10, 15),
        num_tasks=cfg["num_eval_tasks"],
    )

    # 3. RL² evaluation
    print("\n  Evaluating RL² on rare disease episodes...")
    rl2_results = evaluator.evaluate_rl2(
        rl2_trainer=rl2_trainer,
        test_gen=data["rl2_test_gen"],
        num_episodes=cfg["num_eval_episodes"],
    )

    # 4. Per-disease MAML accuracy
    print("\n  Computing per-disease MAML accuracy...")
    per_disease = evaluator.per_disease_accuracy(
        maml_trainer=maml_trainer,
        test_groups=data["test_groups"],
        k_shot=10,
    )

    # 5. Print full report
    evaluator.print_full_report(
        maml_results=maml_results,
        rl2_results=rl2_results,
        baseline_results=baseline_results,
        per_disease_results=per_disease,
    )

    return baseline_results, maml_results, rl2_results, per_disease


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5: VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def phase5_visualise(cfg, baseline_results, maml_results, rl2_results,
                     per_disease, maml_train_hist, rl2_train_hist):
    print("\n" + "█" * 70)
    print("  PHASE 5: GENERATING VISUALISATIONS")
    print("█" * 70)

    # Memory effect plot
    if rl2_results.get("step_accuracy"):
        plot_rl2_memory_effect(
            rl2_results["step_accuracy"],
            n_way=cfg["n_way"]
        )

    # Comparison chart
    plot_comparison(
        baseline_results=baseline_results,
        maml_results=maml_results,
        rl2_acc=rl2_results["mean_accuracy"],
        k_shots=[k for k in [5, 10, 15] if k in maml_results],
    )

    # Per-disease accuracy
    if per_disease:
        plot_per_disease_accuracy(per_disease)

    # Summary dashboard
    plot_summary_dashboard(
        maml_results=maml_results,
        rl2_results=rl2_results,
        train_history_maml=maml_train_hist,
        train_history_rl2=rl2_train_hist,
    )

    print(f"\n  All plots saved to → ./plots/")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS TO JSON
# ─────────────────────────────────────────────────────────────────────────────
def save_results(cfg, baseline_results, maml_results, rl2_results, per_disease):
    os.makedirs("results", exist_ok=True)
    results = {
        "config": cfg,
        "baseline": {str(k): list(v) for k, v in baseline_results.items()},
        "maml":     {str(k): list(v) for k, v in maml_results.items()},
        "rl2": {
            "mean_accuracy": rl2_results["mean_accuracy"],
            "std_accuracy":  rl2_results["std_accuracy"],
            "mean_reward":   rl2_results["mean_reward"],
        },
        "per_disease": {d: list(v) for d, v in per_disease.items()},
    }
    path = "results/evaluation_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Rare Disease Diagnosis — MAML + RL²"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Run with reduced iterations for quick testing")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate from saved checkpoints")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip slow standard baseline evaluation")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Config
    cfg = get_config(quick=args.quick)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("\n" + "█" * 70)
    print("  RARE DISEASE DIAGNOSIS USING META-RL (MAML + RL²)")
    print("  Project Guide: Dr. Lakhan Dev Sharma")
    print("█" * 70)
    print(f"\n  Mode: {'QUICK TEST' if args.quick else 'FULL TRAINING'}")

    # ── Phase 1: Data ────────────────────────────────────────────────────────
    data = phase1_data(cfg)

    if not args.eval_only:
        # ── Phase 2: MAML ────────────────────────────────────────────────────
        maml_trainer, maml_train_hist, maml_val_hist = phase2_maml(cfg, data, device)

        # ── Phase 3: RL² ─────────────────────────────────────────────────────
        rl2_trainer, rl2_train_hist, rl2_val_hist = phase3_rl2(cfg, data, device)

    else:
        print("\n  Loading saved checkpoints...")
        from maml_trainer import MAMLTrainer

        model = DiagnosticClassifier(data["num_features"], cfg["n_way"])
        maml_trainer = MAMLTrainer(model, device=device)
        if os.path.exists(cfg["maml_ckpt"]):
            ckpt = torch.load(cfg["maml_ckpt"], map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            print(f"  MAML checkpoint loaded (iter {ckpt['iteration']})")

        agent = RL2DiagnosticAgent(data["num_features"], cfg["n_way"],
                                   cfg["rl2_gru_hidden"], cfg["rl2_gru_layers"])
        rl2_trainer = RL2Trainer(agent, device=device)
        if os.path.exists(cfg["rl2_ckpt"]):
            ckpt = torch.load(cfg["rl2_ckpt"], map_location=device, weights_only=False)
            agent.load_state_dict(ckpt["model_state"])
            print(f"  RL² checkpoint loaded (ep {ckpt['episode']})")

        maml_train_hist = {"loss": [], "acc": []}
        rl2_train_hist = {"episode_reward": [], "episode_accuracy": [],
                          "policy_loss": [], "entropy": [], "value_loss": []}

    # ── Phase 4: Evaluation ───────────────────────────────────────────────────
    if args.no_baseline:
        # Dummy baseline
        baseline_results = {k: (np.random.uniform(20, 40), 5) for k in [5, 10, 15]}
    else:
        baseline_results = None  # computed inside phase4

    baseline_results, maml_results, rl2_results, per_disease = \
        phase4_evaluation(cfg, data, maml_trainer, rl2_trainer, device)

    # ── Phase 5: Visualisation ─────────────────────────────────────────────
    phase5_visualise(cfg, baseline_results, maml_results, rl2_results,
                     per_disease, maml_train_hist, rl2_train_hist)

    # Save results
    save_results(cfg, baseline_results, maml_results, rl2_results, per_disease)

    print("\n" + "█" * 70)
    print("  COMPLETE! All outputs in ./plots/ and ./results/")
    print("█" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE LOGIC (API TARGET)
# ─────────────────────────────────────────────────────────────────────────────

# Global state for fast API calls
_api_model = None
_api_symptom_list = None
_api_disease_list = None
_api_num_features = None

def init_inference_model(quick=False):
    """
    Initialises the cached model and dataset for extremely fast API queries.
    """
    global _api_model, _api_symptom_list, _api_disease_list, _api_num_features
    if _api_model is not None:
        return
        
    print("[API Setup] Initialising Meta-Learning Inference Model...")
    cfg = get_config(quick=quick)
    from data_preprocessing import load_dataset, build_feature_matrix, DATA_PATH
    import os
    
    df = load_dataset(DATA_PATH)
    _, _, _api_symptom_list, _api_disease_list, _ = build_feature_matrix(df)
    _api_num_features = len(_api_symptom_list)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _api_model = DiagnosticClassifier(num_features=_api_num_features, n_way=cfg["n_way"])
    
    ckpt_path = os.path.join(os.path.dirname(__file__), cfg["maml_ckpt"])
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        _api_model.load_state_dict(ckpt["model_state"])
        _api_model.to(device)
        _api_model.eval()
        print(f"[API Setup] Inference model successfully loaded from {ckpt_path}")
    else:
        print(f"[API Setup] Warning: Checkpoint {ckpt_path} not found.")

def predict_disease(symptoms):
    """
    Takes a list of symptoms (strings), processes them, and returns a fast prediction dict.
    Must be fast and non-blocking for the API Server.
    """
    if _api_model is None:
        init_inference_model()
        
    features = np.zeros((1, _api_num_features), dtype=np.float32)
    valid_symptoms = []
    
    # Process string symptoms to 1D binary vector
    if isinstance(symptoms, list):
        for s in symptoms:
            if s in _api_symptom_list:
                features[0, _api_symptom_list.index(s)] = 1.0
                valid_symptoms.append(s)
                
    device = next(_api_model.parameters()).device
    x_tensor = torch.tensor(features).to(device)
    
    with torch.no_grad():
        log_probs = _api_model(x_tensor)
        probs = torch.exp(log_probs).squeeze(0).cpu().numpy()
        pred_idx = np.argmax(probs)
        
    n_way = _api_model.n_way
    available_diseases = _api_disease_list[:n_way]
    
    predicted_disease = available_diseases[pred_idx]
    confidence_scores = {available_diseases[i]: float(probs[i]) for i in range(n_way)}
    
    return {
        'predicted_disease': predicted_disease,
        'confidence': float(probs[pred_idx]),
        'confidence_scores': confidence_scores,
        'valid_symptoms_processed': valid_symptoms
    }
    
def get_available_symptoms():
    if _api_symptom_list is None:
        init_inference_model()
    return _api_symptom_list


if __name__ == "__main__":
    main()
