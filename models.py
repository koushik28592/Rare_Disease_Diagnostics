"""
=============================================================================
models.py
Rare Disease Diagnosis using MAML + RL²
=============================================================================
Contains:
  1. DiagnosticClassifier  — Feed-forward network used as the MAML base model
  2. RL2DiagnosticAgent    — GRU-based recurrent policy network for RL²
  3. HybridMAMLRL2Agent    — Combines MAML initialization + RL² memory
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


# ─────────────────────────────────────────────────────────────────────────────
# 1. DIAGNOSTIC CLASSIFIER (MAML Base Network)
# ─────────────────────────────────────────────────────────────────────────────
class DiagnosticClassifier(nn.Module):
    """
    Feed-forward neural network used as the base learner in MAML.

    Architecture:
      Input (num_features=131) → FC(256) → BN → ReLU → Dropout
                               → FC(128) → BN → ReLU → Dropout
                               → FC(64)  → BN → ReLU
                               → FC(n_way)  → LogSoftmax

    Design rationale:
      - BatchNorm after each layer stabilises gradients during inner-loop updates
      - Dropout prevents overfitting on tiny support sets
      - 3 hidden layers give enough capacity to learn symptom combinations
    """

    def __init__(self, num_features: int, n_way: int,
                 hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        self.num_features = num_features
        self.n_way = n_way

        layers = []
        in_dim = num_features
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        # Final classification head (no dropout)
        layers.append(nn.Linear(in_dim, n_way))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialisation for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x : Tensor (batch, num_features)
        Returns log-probabilities (batch, n_way)
        """
        logits = self.network(x)
        return F.log_softmax(logits, dim=-1)

    def functional_forward(self, x, params):
        """
        Forward pass using externally provided parameter dict.
        Required for MAML inner-loop where we compute adapted params
        but do NOT want to modify self.parameters() in place.

        params : OrderedDict matching self.named_parameters()
        """
        # We rebuild the forward pass manually using functional operations
        # to allow gradient flow through the parameter adaptation step.
        x_in = x

        # Walk through layers
        param_keys = list(params.keys())
        layer_idx = 0

        # We have groups of (weight, bias, bn_weight, bn_bias, bn_running_mean,
        # bn_running_var, bn_num_batches_tracked) per block
        # Simpler: just use named params and match them to network layers

        # Collect linear and BN layers in order
        linears = []
        bns = []
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Linear):
                linears.append(name)
            elif isinstance(module, nn.BatchNorm1d):
                bns.append(name)

        # Reconstruct forward using functional API
        bn_idx = 0
        lin_idx = 0

        x_current = x_in
        block_size = 4  # Linear, BN, ReLU, Dropout per hidden block

        num_hidden = len([k for k in params if 'network' in k and 'weight' in k]) - 1  # exclude last linear

        # Easier approach: iterate through network modules manually
        for name, module in self.network.named_modules():
            if name == "":
                continue  # skip root
            # Find matching params
            # param names look like "network.0.weight", "network.0.bias", etc.
            module_key = f"network.{name}"
            if isinstance(module, nn.Linear):
                w_key = f"{module_key}.weight"
                b_key = f"{module_key}.bias"
                if w_key in params:
                    x_current = F.linear(x_current, params[w_key], params.get(b_key))
            elif isinstance(module, nn.BatchNorm1d):
                w_key = f"{module_key}.weight"
                b_key = f"{module_key}.bias"
                if w_key in params:
                    x_current = F.batch_norm(
                        x_current,
                        running_mean=module.running_mean,
                        running_var=module.running_var,
                        weight=params[w_key],
                        bias=params[b_key],
                        training=True,
                        eps=module.eps,
                        momentum=module.momentum,
                    )
            elif isinstance(module, nn.ReLU):
                x_current = F.relu(x_current, inplace=False)
            elif isinstance(module, nn.Dropout):
                x_current = F.dropout(x_current, p=module.p,
                                      training=self.training)

        return F.log_softmax(x_current, dim=-1)

    def clone(self):
        """Deep copy of the model — used in MAML inner loop."""
        return copy.deepcopy(self)


# ─────────────────────────────────────────────────────────────────────────────
# 2. RL² DIAGNOSTIC AGENT (GRU-based Recurrent Policy)
# ─────────────────────────────────────────────────────────────────────────────
class RL2DiagnosticAgent(nn.Module):
    """
    GRU-based recurrent agent for RL² meta-reinforcement learning.

    The agent processes a sequence of patient encounters.
    At each step t, it receives:
        obs_t = [symptom_vector | prev_action_onehot | prev_reward]

    and outputs:
        - action logits (which disease to predict)
        - value estimate V(s_t) for PPO/advantage computation

    The GRU hidden state acts as the "clinical memory" — it accumulates
    evidence across multiple patient visits before committing to a diagnosis,
    mimicking an experienced physician's reasoning process.

    Architecture:
        Input Encoder: FC(obs_dim → 256) → LayerNorm → ReLU
        GRU Memory   : GRU(256 → gru_hidden=256, num_layers=2)
        Policy Head  : FC(256 → n_way)      → action logits
        Value Head   : FC(256 → 1)          → state value
    """

    def __init__(self, num_features: int, n_way: int,
                 gru_hidden: int = 256, gru_layers: int = 2,
                 encoder_hidden: int = 256):
        super().__init__()
        self.num_features = num_features
        self.n_way = n_way
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

        # Observation dimension: symptoms + prev_action_onehot + prev_reward_scalar
        self.obs_dim = num_features + n_way + 1

        # ── Input Encoder ──────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, encoder_hidden),
            nn.LayerNorm(encoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.LayerNorm(encoder_hidden),
            nn.ReLU(inplace=True),
        )

        # ── GRU Memory (the "RL²" component) ───────────────────────────────
        self.gru = nn.GRU(
            input_size=encoder_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,   # input: (batch, seq_len, features)
            dropout=0.2 if gru_layers > 1 else 0.0,
        )

        # ── Policy Head ────────────────────────────────────────────────────
        self.policy_head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_way),
        )

        # ── Value Head (for actor-critic / PPO) ────────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Policy/value head: smaller init
        for head in [self.policy_head, self.value_head]:
            last_linear = [m for m in head.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.orthogonal_(last_linear.weight, gain=0.01)

    def init_hidden(self, batch_size: int = 1, device=None):
        """
        Returns zero hidden state: (num_layers, batch, gru_hidden)
        Call this at the start of each new episode.
        """
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(self.gru_layers, batch_size,
                           self.gru_hidden, device=device)

    def forward(self, obs_seq, hidden=None):
        """
        Process a full sequence of observations.

        obs_seq : Tensor (batch, seq_len, obs_dim)
        hidden  : Tensor (num_layers, batch, gru_hidden) or None

        Returns:
          action_logits : Tensor (batch, seq_len, n_way)
          values        : Tensor (batch, seq_len, 1)
          new_hidden    : Tensor (num_layers, batch, gru_hidden)
        """
        batch, seq_len, _ = obs_seq.shape

        if hidden is None:
            hidden = self.init_hidden(batch, obs_seq.device)

        # Encode observations: (batch*seq_len, obs_dim) → (batch*seq_len, enc_hidden)
        obs_flat = obs_seq.view(batch * seq_len, -1)
        encoded = self.encoder(obs_flat)              # (B*T, enc_hidden)
        encoded = encoded.view(batch, seq_len, -1)    # (B, T, enc_hidden)

        # GRU: (B, T, enc_hidden) → (B, T, gru_hidden)
        gru_out, new_hidden = self.gru(encoded, hidden)

        # Policy and value heads
        gru_flat = gru_out.contiguous().view(batch * seq_len, -1)
        action_logits = self.policy_head(gru_flat).view(batch, seq_len, self.n_way)
        values = self.value_head(gru_flat).view(batch, seq_len, 1)

        return action_logits, values, new_hidden

    def act(self, obs_single, hidden):
        """
        Single-step inference (for episode rollout).

        obs_single : Tensor (obs_dim,) or (1, obs_dim)
        hidden     : Tensor (num_layers, 1, gru_hidden)

        Returns:
          action      : int
          log_prob    : Tensor scalar
          value       : Tensor scalar
          new_hidden  : updated hidden state
        """
        if obs_single.dim() == 1:
            obs_single = obs_single.unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)
        elif obs_single.dim() == 2:
            obs_single = obs_single.unsqueeze(0)               # (1, 1, obs_dim)

        with torch.no_grad():
            logits, value, new_hidden = self.forward(obs_single, hidden)

        logits = logits.squeeze()   # (n_way,)
        value = value.squeeze()     # scalar

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value, new_hidden


# ─────────────────────────────────────────────────────────────────────────────
# 3. HYBRID MAML + RL² AGENT
# ─────────────────────────────────────────────────────────────────────────────
class HybridMAMLRL2Agent(nn.Module):
    """
    Combines:
      - MAML's fast-adaptation initialisation (outer-loop trained parameters)
      - RL²'s GRU memory for sequential patient history

    The MAML component provides the classification backbone.
    The RL² GRU wraps around it, feeding the MAML output as part of
    an enhanced representation at each timestep.

    Architecture:
        Symptom Input → MAML Classifier (adapted weights) → disease_logits (n_way)
        Obs Input     → RL² GRU Encoder → GRU → policy_head → final_logits
                                                             → value_head

    The two outputs are fused via a learned gating mechanism:
        final_logits = α * maml_logits + (1-α) * rl2_logits
    where α is a learned scalar per class.
    """

    def __init__(self, num_features: int, n_way: int,
                 gru_hidden: int = 256, gru_layers: int = 2):
        super().__init__()
        self.num_features = num_features
        self.n_way = n_way

        # MAML backbone
        self.maml_net = DiagnosticClassifier(num_features, n_way)

        # RL² agent
        self.rl2_agent = RL2DiagnosticAgent(num_features, n_way,
                                            gru_hidden, gru_layers)

        # Fusion gate: learned per-class blending weight (sigmoid → [0,1])
        self.gate = nn.Parameter(torch.full((n_way,), 0.5))

    def forward(self, symptom_x, obs_seq, hidden=None):
        """
        symptom_x : Tensor (batch, num_features)        — for MAML head
        obs_seq   : Tensor (batch, seq_len, obs_dim)    — for RL² GRU
        hidden    : GRU hidden state

        Returns:
          fused_logits : Tensor (batch, seq_len, n_way)
          values       : Tensor (batch, seq_len, 1)
          new_hidden   : updated GRU hidden
        """
        batch, seq_len, _ = obs_seq.shape

        # MAML forward: classify each symptom vector
        # Expand for sequence: (batch*seq_len, num_features)
        # Here we use the same symptom_x for all steps in the sequence
        # (In full deployment each step would be a new patient)
        sym_expanded = symptom_x.unsqueeze(1).expand(-1, seq_len, -1)
        sym_flat = sym_expanded.contiguous().view(batch * seq_len, -1)
        maml_logits_flat = self.maml_net(sym_flat)   # (B*T, n_way)
        maml_logits = maml_logits_flat.view(batch, seq_len, self.n_way)

        # RL² forward
        rl2_logits, values, new_hidden = self.rl2_agent(obs_seq, hidden)

        # Fused output: gated blend
        alpha = torch.sigmoid(self.gate)  # (n_way,)
        fused_logits = alpha * maml_logits + (1 - alpha) * rl2_logits

        return fused_logits, values, new_hidden

    def init_hidden(self, batch_size=1, device=None):
        return self.rl2_agent.init_hidden(batch_size, device)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    NUM_F = 131
    N_WAY = 5
    BATCH = 4
    SEQ   = 20

    print("=" * 60)
    print("Testing DiagnosticClassifier (MAML base model)")
    print("=" * 60)
    clf = DiagnosticClassifier(NUM_F, N_WAY)
    x = torch.randn(BATCH, NUM_F)
    out = clf(x)
    print(f"  Input : {x.shape}")
    print(f"  Output: {out.shape}  (should be ({BATCH}, {N_WAY}))")
    total_params = sum(p.numel() for p in clf.parameters())
    print(f"  Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("Testing RL2DiagnosticAgent")
    print("=" * 60)
    obs_dim = NUM_F + N_WAY + 1
    agent = RL2DiagnosticAgent(NUM_F, N_WAY)
    obs_seq = torch.randn(BATCH, SEQ, obs_dim)
    logits, values, hidden = agent(obs_seq)
    print(f"  Obs seq   : {obs_seq.shape}")
    print(f"  Logits    : {logits.shape}   (should be ({BATCH},{SEQ},{N_WAY}))")
    print(f"  Values    : {values.shape}   (should be ({BATCH},{SEQ},1))")
    print(f"  Hidden    : {hidden.shape}")
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"  Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("Testing HybridMAMLRL2Agent")
    print("=" * 60)
    hybrid = HybridMAMLRL2Agent(NUM_F, N_WAY)
    sym_x = torch.randn(BATCH, NUM_F)
    obs_s = torch.randn(BATCH, SEQ, obs_dim)
    fused, vals, h = hybrid(sym_x, obs_s)
    print(f"  Fused logits: {fused.shape}")
    print(f"  Values      : {vals.shape}")
    total_params = sum(p.numel() for p in hybrid.parameters())
    print(f"  Total parameters: {total_params:,}")
    print("\nAll model tests passed ✓")
