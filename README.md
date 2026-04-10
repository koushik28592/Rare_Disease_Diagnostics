# Rare Disease Diagnosis using MAML + RL²
**Project Guide: Dr. Lakhan Dev Sharma | School of Electronics Engineering (SENSE)**

---

## Project Overview

This project implements a **Hybrid Meta-Reinforcement Learning system** for rare disease diagnosis
that combines:

- **MAML** (Model-Agnostic Meta-Learning) — for fast adaptation with few patient examples
- **RL²** (Reinforcement Learning Squared) — for sequential clinical reasoning via GRU memory

The system can accurately diagnose rare diseases using only **5–10 patient records**, compared to
the thousands required by standard deep learning.

---

## Dataset

**DiseaseAndSymptoms.csv**
| Property | Value |
|---|---|
| Total Records | 4,920 |
| Diseases | 41 unique |
| Records per disease | 120 |
| Unique symptoms | 131 |
| Symptom columns | Symptom_1 to Symptom_17 |

**Task split:**
- Meta-train: 35 diseases (common diseases seen during meta-training)
- Meta-test: 6 diseases (held out as "rare" — agent has never seen these)

---

## File Structure

```
rare_disease_diagnosis/
│
├── data_preprocessing.py   # Dataset loading, feature engineering, task samplers
├── models.py               # DiagnosticClassifier, RL2Agent, HybridAgent
├── maml_trainer.py         # MAML outer/inner loop training
├── rl2_trainer.py          # RL² PPO training loop
├── evaluation.py           # Evaluation suite and comparison table
├── visualisation.py        # All plots and figures
├── main.py                 # Full training pipeline
├── demo.py                 # Quick demo with inference walkthrough
├── requirements.txt
│
├── DiseaseAndSymptoms.csv  # Dataset (place here)
├── checkpoints/            # Saved model weights
├── plots/                  # Generated figures
└── results/                # Evaluation JSON
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Quick demo (reduced iterations, ~5-10 min)
python demo.py

# Full training (300 MAML iterations + 1000 RL² episodes)
python main.py

# Evaluation only (from saved checkpoints)
python main.py --eval-only
```

---

## Architecture

### MAML (Outer + Inner Loop)

```
Meta-Training (Outer Loop):
  For each iteration:
    Sample 8 disease tasks from 35 common diseases
    For each task:
      INNER LOOP: 5 gradient steps on support set (10 examples)
      Compute query loss with adapted weights θ'
    Outer update: θ ← θ - β * ∇_θ Σ L_query(f_θ'_i)

Meta-Testing (Rare Disease):
  New disease arrives with 5-10 examples
  Inner loop adapts in 1-2 gradient steps → high accuracy
```

### RL² (GRU Memory Agent + PPO)

```
Episode structure (T=20 patient steps):
  h_0 = zeros (GRU hidden state)
  For t = 1..T:
    obs_t = [symptom_vector | prev_action_onehot | prev_reward]
    (action_t, value_t) = GRU_policy(obs_t, h_{t-1})
    reward_t = +1 if correct, -0.5 if wrong
    h_t = GRU_update(h_{t-1}, obs_t)

PPO Update after each episode:
  Compute GAE advantages
  Clip surrogate objective
  Update GRU weights
```

### Hybrid System

```
MAML provides: optimised initial weights θ*
RL² provides:  GRU sequential memory

Combined:
  symptom_x → MAML_classifier → maml_logits
  obs_seq   → RL2_GRU        → rl2_logits
  fused = sigmoid(α) * maml_logits + (1-sigmoid(α)) * rl2_logits
```

---

## Expected Results

| Method | Examples Needed | Accuracy |
|---|---|---|
| Standard AI | 50+ | ~35% |
| MAML (5-shot) | 5 | ~65% |
| MAML (10-shot) | 10 | ~75% |
| MAML (15-shot) | 15 | ~82% |
| RL² (sequential) | stream | ~70% |

Key finding: MAML achieves **2× the accuracy** of standard AI with **10× fewer** training examples.

---

## References

1. Finn, Abbeel, Levine — "Model-Agnostic Meta-Learning for Fast Adaptation" (ICML 2017)
2. Duan et al. — "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning" (arXiv 2016)
3. Wang et al. — "A Comprehensive Survey on Meta-Learning" (IEEE TPAMI 2024)
4. Beck et al. — "A Comprehensive Survey of Meta-Reinforcement Learning" (IEEE TNNLS 2023)
