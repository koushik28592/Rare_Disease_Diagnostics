"""
=============================================================================
data_preprocessing.py
Rare Disease Diagnosis using MAML + RL²
=============================================================================
Handles:
  - Loading the DiseaseAndSymptoms.csv dataset
  - Building a binary symptom feature matrix (one-hot encoding)
  - Splitting diseases into meta-train / meta-test (rare disease simulation)
  - Generating few-shot task batches for MAML
  - Generating sequential patient episode streams for RL²
=============================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import random
import torch

# ─────────────────────────────────────────────
# SEED FOR REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DATA_PATH = r"C:\Users\koush\Downloads\archive (1)\DiseaseAndSymptoms.csv"

# How many diseases to hold out as "rare" at test time
NUM_RARE_DISEASES = 6
# Few-shot support set size (simulating real-world rare disease scarcity)
N_SHOT = 10
# Query set size per task
N_QUERY = 15


# ─────────────────────────────────────────────
# LOAD & CLEAN DATASET
# ─────────────────────────────────────────────
def load_dataset(path=DATA_PATH):
    """
    Load the CSV, clean column names and symptom strings.
    Returns a DataFrame with Disease + symptom columns.
    """
    df = pd.read_csv(path)

    # Standardise column names
    df.columns = [c.strip() for c in df.columns]

    # Strip whitespace from all symptom cells
    sym_cols = [c for c in df.columns if c.startswith("Symptom_")]
    for col in sym_cols:
        df[col] = df[col].fillna("").str.strip()

    # Strip whitespace from disease names
    df["Disease"] = df["Disease"].str.strip()

    print(f"[DataLoader] Loaded {len(df)} records | "
          f"{df['Disease'].nunique()} diseases | "
          f"{len(sym_cols)} symptom columns")
    return df


# ─────────────────────────────────────────────
# BUILD BINARY SYMPTOM FEATURE MATRIX
# ─────────────────────────────────────────────
def build_feature_matrix(df):
    """
    Converts the wide symptom columns into a binary feature matrix.
    Each unique symptom becomes one dimension (131-dim vector).

    Returns:
      X         : np.ndarray (N, num_symptoms) — binary features
      y         : np.ndarray (N,)              — integer disease labels
      symptom_list : list of symptom names
      disease_list : list of disease names (label order)
      label_enc    : fitted LabelEncoder for diseases
    """
    sym_cols = [c for c in df.columns if c.startswith("Symptom_")]

    # Collect all unique symptom names
    all_symptoms = set()
    for col in sym_cols:
        all_symptoms.update(df[col].unique())
    all_symptoms.discard("")  # remove empty string
    symptom_list = sorted(list(all_symptoms))
    sym_to_idx = {s: i for i, s in enumerate(symptom_list)}

    # Build binary matrix
    X = np.zeros((len(df), len(symptom_list)), dtype=np.float32)
    for row_idx, row in df.iterrows():
        for col in sym_cols:
            sym = row[col]
            if sym and sym in sym_to_idx:
                X[row_idx, sym_to_idx[sym]] = 1.0

    # Encode disease labels
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df["Disease"].values)
    disease_list = list(label_enc.classes_)

    print(f"[FeatureBuilder] Feature matrix: {X.shape} | "
          f"Unique symptoms: {len(symptom_list)} | "
          f"Classes: {len(disease_list)}")
    return X, y, symptom_list, disease_list, label_enc


# ─────────────────────────────────────────────
# SPLIT DISEASES INTO META-TRAIN / META-TEST
# ─────────────────────────────────────────────
def split_diseases(disease_list, num_rare=NUM_RARE_DISEASES, seed=SEED):
    """
    Randomly hold out `num_rare` diseases as unseen rare diseases.
    The rest are used for meta-training.

    Returns:
      train_diseases : list of disease names for meta-training
      test_diseases  : list of rare disease names for meta-testing
    """
    rng = random.Random(seed)
    shuffled = disease_list.copy()
    rng.shuffle(shuffled)
    test_diseases = shuffled[:num_rare]
    train_diseases = shuffled[num_rare:]

    print(f"\n[Split] META-TRAIN diseases ({len(train_diseases)}):")
    for d in sorted(train_diseases):
        print(f"        • {d}")
    print(f"\n[Split] META-TEST (RARE) diseases ({len(test_diseases)}):")
    for d in sorted(test_diseases):
        print(f"        ★ {d}")
    return train_diseases, test_diseases


# ─────────────────────────────────────────────
# HELPER: group indices by disease label
# ─────────────────────────────────────────────
def group_by_disease(X, y, disease_list, label_enc):
    """
    Returns a dict: {disease_name: (X_subset, y_subset)}
    where y_subset are the local integer labels (0, 1, 2...) within the group.
    For MAML tasks we only need the binary disease-level label.
    """
    groups = {}
    for disease in disease_list:
        cls_idx = label_enc.transform([disease])[0]
        mask = (y == cls_idx)
        groups[disease] = X[mask]
    return groups


# ─────────────────────────────────────────────
# MAML TASK SAMPLER
# ─────────────────────────────────────────────
class MAMLTaskSampler:
    """
    Samples N-way K-shot classification tasks for MAML.

    Each task:
      - Randomly selects `n_way` diseases from the available pool
      - For each disease, samples `k_shot` support examples + `n_query` query examples
      - Returns tensors ready for the MAML inner loop
    """

    def __init__(self, groups, n_way=5, k_shot=N_SHOT, n_query=N_QUERY):
        """
        groups  : dict {disease_name: X_array}
        n_way   : number of classes per task
        k_shot  : support set size per class
        n_query : query set size per class
        """
        self.groups = groups
        self.disease_names = list(groups.keys())
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

    def sample_task(self):
        """
        Returns:
          support_x : Tensor (n_way * k_shot, num_features)
          support_y : Tensor (n_way * k_shot,) — local labels 0..n_way-1
          query_x   : Tensor (n_way * n_query, num_features)
          query_y   : Tensor (n_way * n_query,) — local labels 0..n_way-1
          selected_diseases : list of disease names for this task
        """
        selected = random.sample(self.disease_names, self.n_way)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for local_label, disease in enumerate(selected):
            X_pool = self.groups[disease]
            total_needed = self.k_shot + self.n_query

            if len(X_pool) < total_needed:
                # Sample with replacement if pool too small
                indices = np.random.choice(len(X_pool), total_needed, replace=True)
            else:
                indices = np.random.choice(len(X_pool), total_needed, replace=False)

            support_indices = indices[:self.k_shot]
            query_indices = indices[self.k_shot:]

            support_x.append(X_pool[support_indices])
            support_y.extend([local_label] * self.k_shot)

            query_x.append(X_pool[query_indices])
            query_y.extend([local_label] * self.n_query)

        support_x = torch.FloatTensor(np.vstack(support_x))
        support_y = torch.LongTensor(support_y)
        query_x = torch.FloatTensor(np.vstack(query_x))
        query_y = torch.LongTensor(query_y)

        return support_x, support_y, query_x, query_y, selected

    def sample_batch(self, num_tasks):
        """Sample a batch of tasks for one meta-training iteration."""
        return [self.sample_task() for _ in range(num_tasks)]


# ─────────────────────────────────────────────
# RL² EPISODE GENERATOR
# ─────────────────────────────────────────────
class RL2EpisodeGenerator:
    """
    Generates sequential patient episodes for RL².

    Each episode = a sequence of patient encounters for ONE disease task.
    The agent sees patients one at a time, predicts the disease,
    receives a reward, and updates its GRU hidden state.

    Episode structure per step t:
      observation : (symptom_vector || prev_action_onehot || prev_reward)
                    Shape: (num_symptoms + n_way + 1,)
      action      : integer class label (local, 0..n_way-1)
      reward      : +1 if correct, -0.5 if wrong (shaped reward)
    """

    def __init__(self, groups, n_way=5, episode_length=20):
        """
        groups         : dict {disease_name: X_array}
        n_way          : classes per episode
        episode_length : number of patient steps per episode
        """
        self.groups = groups
        self.disease_names = list(groups.keys())
        self.n_way = n_way
        self.episode_length = episode_length

    def generate_episode(self):
        """
        Returns:
          observations : Tensor (episode_length, obs_dim)
          actions      : Tensor (episode_length,) — true labels
          rewards      : Tensor (episode_length,) — shaped rewards
          selected_diseases : list of disease names
        """
        selected = random.sample(self.disease_names, self.n_way)
        disease_to_local = {d: i for i, d in enumerate(selected)}

        num_features = next(iter(self.groups.values())).shape[1]
        obs_dim = num_features + self.n_way + 1  # symptom + prev_action_oh + prev_reward

        observations = []
        true_actions = []
        rewards_list = []

        prev_action_oh = np.zeros(self.n_way, dtype=np.float32)
        prev_reward = np.array([0.0], dtype=np.float32)

        for step in range(self.episode_length):
            # Sample a random disease and one patient from it
            disease = random.choice(selected)
            local_label = disease_to_local[disease]
            X_pool = self.groups[disease]
            patient = X_pool[np.random.randint(len(X_pool))]  # (num_features,)

            # Construct observation: [symptoms | prev_action | prev_reward]
            obs = np.concatenate([patient, prev_action_oh, prev_reward])
            observations.append(obs)
            true_actions.append(local_label)

            # Reward: +1 correct, -0.5 wrong (will be computed during training)
            # Here we store ground truth; actual reward depends on model's prediction
            # We store shaped reward based on oracle for supervised pre-training
            rewards_list.append(1.0)  # placeholder; overridden during RL training

            # Update prev action (oracle — for teacher forcing during initial training)
            prev_action_oh = np.zeros(self.n_way, dtype=np.float32)
            prev_action_oh[local_label] = 1.0
            prev_reward = np.array([1.0], dtype=np.float32)

        observations = torch.FloatTensor(np.array(observations))  # (T, obs_dim)
        true_actions = torch.LongTensor(true_actions)              # (T,)
        rewards_tensor = torch.FloatTensor(rewards_list)           # (T,)

        return observations, true_actions, rewards_tensor, selected, num_features


# ─────────────────────────────────────────────
# FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
def prepare_all(data_path=DATA_PATH, num_rare=NUM_RARE_DISEASES,
                n_way=5, k_shot=N_SHOT, n_query=N_QUERY, episode_length=20):
    """
    Master function: runs the full preprocessing pipeline.

    Returns a dict with everything needed for training:
      {
        'X', 'y',
        'symptom_list', 'disease_list', 'label_enc',
        'train_diseases', 'test_diseases',
        'train_groups', 'test_groups',
        'maml_train_sampler', 'maml_test_sampler',
        'rl2_train_gen', 'rl2_test_gen',
        'num_features', 'n_way'
      }
    """
    df = load_dataset(data_path)
    X, y, symptom_list, disease_list, label_enc = build_feature_matrix(df)
    train_diseases, test_diseases = split_diseases(disease_list, num_rare)

    train_groups = group_by_disease(X, y, train_diseases, label_enc)
    test_groups = group_by_disease(X, y, test_diseases, label_enc)

    maml_train_sampler = MAMLTaskSampler(train_groups, n_way, k_shot, n_query)
    maml_test_sampler = MAMLTaskSampler(test_groups,
                                        n_way=min(n_way, len(test_diseases)),
                                        k_shot=k_shot, n_query=n_query)

    rl2_train_gen = RL2EpisodeGenerator(train_groups, n_way, episode_length)
    rl2_test_gen = RL2EpisodeGenerator(test_groups,
                                       n_way=min(n_way, len(test_diseases)),
                                       episode_length=episode_length)

    num_features = X.shape[1]

    return {
        "X": X, "y": y,
        "symptom_list": symptom_list,
        "disease_list": disease_list,
        "label_enc": label_enc,
        "train_diseases": train_diseases,
        "test_diseases": test_diseases,
        "train_groups": train_groups,
        "test_groups": test_groups,
        "maml_train_sampler": maml_train_sampler,
        "maml_test_sampler": maml_test_sampler,
        "rl2_train_gen": rl2_train_gen,
        "rl2_test_gen": rl2_test_gen,
        "num_features": num_features,
        "n_way": n_way,
    }


if __name__ == "__main__":
    data = prepare_all()
    print("\n[Test] Sampling one MAML task...")
    sx, sy, qx, qy, diseases = data["maml_train_sampler"].sample_task()
    print(f"  Support X: {sx.shape}, Support Y: {sy.shape}")
    print(f"  Query   X: {qx.shape}, Query   Y: {qy.shape}")
    print(f"  Diseases : {diseases}")

    print("\n[Test] Generating one RL² episode...")
    obs, acts, rews, diseases, nf = data["rl2_train_gen"].generate_episode()
    print(f"  Obs shape   : {obs.shape}")
    print(f"  Actions     : {acts.shape}")
    print(f"  Diseases    : {diseases}")
    print(f"  Obs dim     : {obs.shape[1]} = {nf} symptoms + {data['n_way']} action_oh + 1 reward")
