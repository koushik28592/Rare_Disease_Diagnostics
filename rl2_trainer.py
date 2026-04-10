"""
=============================================================================
rl2_trainer.py
Rare Disease Diagnosis using MAML + RL²
=============================================================================
Implements RL² (Reinforcement Learning Squared) training using PPO
(Proximal Policy Optimisation) for the GRU-based diagnostic agent.

Key ideas:
  - The GRU agent processes sequences of patient encounters (episodes)
  - Each episode = one disease task with sequential patient observations
  - Reward: +1 if diagnosis correct, -0.5 if wrong
  - PPO objective: surrogate clipped objective + value loss + entropy bonus
  - Hidden state carries diagnostic memory across the entire episode

RL² principle:
  - The OUTER RL (PPO) learns general diagnostic strategy across many diseases
  - The INNER RL is the GRU's hidden state — it "learns" within a single episode
    by updating its memory based on past (symptom, diagnosis, reward) history
  - This two-level learning = RL²

Training loop per episode:
  1. Initialise GRU hidden state h_0 = 0
  2. For t in [1..T]:
       - Observe patient symptoms + prev_action + prev_reward
       - Agent predicts disease (action)
       - Receive reward (+1 correct / -0.5 wrong)
       - Store (obs, action, reward, value, log_prob) in rollout buffer
       - Update hidden state h_t
  3. Compute advantages using GAE
  4. PPO update on the rollout
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os


# ─────────────────────────────────────────────────────────────────────────────
# REWARD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def compute_reward(predicted: int, true_label: int,
                   step: int, episode_length: int) -> float:
    """
    Shaped reward for rare disease diagnosis.

    +1.0  : correct diagnosis
    -0.5  : wrong diagnosis
    +0.2  : bonus for early correct diagnosis (before halfway through episode)
    -0.1  : repeated wrong answers (discourages random exploration at end)
    """
    if predicted == true_label:
        bonus = 0.2 if step < episode_length // 2 else 0.0
        return 1.0 + bonus
    else:
        late_penalty = -0.1 if step > episode_length * 0.75 else 0.0
        return -0.5 + late_penalty


# ─────────────────────────────────────────────────────────────────────────────
# ROLLOUT BUFFER
# ─────────────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    """
    Stores one complete episode's experience for PPO training.
    """

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, gamma=0.99, gae_lambda=0.95):
        """
        Generalised Advantage Estimation (GAE).
        Returns advantages and discounted returns.
        """
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        values_np = np.array([v.item() if torch.is_tensor(v) else v
                               for v in self.values], dtype=np.float32)
        rewards_np = np.array(self.rewards, dtype=np.float32)
        dones_np = np.array(self.dones, dtype=np.float32)

        for t in reversed(range(T)):
            next_val = values_np[t + 1] if t + 1 < T else 0.0
            delta = rewards_np[t] + gamma * next_val * (1 - dones_np[t]) - values_np[t]
            last_gae = delta + gamma * gae_lambda * (1 - dones_np[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values_np
        return advantages, returns

    def get_tensors(self, device):
        """Convert buffer to tensors for PPO update."""
        obs = torch.stack(self.observations).to(device)     # (T, obs_dim)
        acts = torch.tensor(self.actions, dtype=torch.long, device=device)
        old_lps = torch.stack(self.log_probs).to(device)   # (T,)
        return obs, acts, old_lps

    def clear(self):
        self.__init__()


# ─────────────────────────────────────────────────────────────────────────────
# RL² TRAINER
# ─────────────────────────────────────────────────────────────────────────────
class RL2Trainer:
    """
    PPO-based trainer for the RL² GRU diagnostic agent.

    Parameters
    ----------
    agent          : RL2DiagnosticAgent
    lr             : float  — learning rate
    gamma          : float  — discount factor
    gae_lambda     : float  — GAE lambda
    clip_eps       : float  — PPO clip epsilon
    value_coef     : float  — value loss coefficient
    entropy_coef   : float  — entropy bonus coefficient
    ppo_epochs     : int    — PPO update epochs per rollout
    device         : torch.device
    """

    def __init__(self, agent, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
                 ppo_epochs=4, device=None):
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=lr, eps=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=500
        )

        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)

    # ─────────────────────────────────────────────
    # COLLECT ONE EPISODE ROLLOUT
    # ─────────────────────────────────────────────
    def collect_episode(self, episode_gen):
        """
        Runs the agent through one full episode, collecting experience.

        Returns:
          buffer      : RolloutBuffer with full episode data
          ep_reward   : total episode reward
          ep_accuracy : fraction of correct diagnoses in this episode
        """
        obs_seq, true_actions, _, selected_diseases, num_features = \
            episode_gen.generate_episode()

        episode_length = obs_seq.shape[0]
        n_way = self.agent.n_way

        buffer = RolloutBuffer()
        hidden = self.agent.init_hidden(batch_size=1, device=self.device)

        ep_reward = 0.0
        correct_count = 0

        prev_action_oh = np.zeros(n_way, dtype=np.float32)
        prev_reward_val = 0.0

        for t in range(episode_length):
            # Get current patient's symptom vector
            obs_t = obs_seq[t].to(self.device)  # (obs_dim,)

            # Reconstruct obs with actual prev_action (not oracle)
            # We rebuild obs_t from the symptom portion + our prev info
            sym_t = obs_t[:num_features]
            obs_reconstructed = torch.cat([
                sym_t,
                torch.FloatTensor(prev_action_oh).to(self.device),
                torch.FloatTensor([prev_reward_val]).to(self.device),
            ])

            # Agent decision
            action, log_prob, value, hidden = self.agent.act(
                obs_reconstructed, hidden
            )

            # Compute reward
            true_label = true_actions[t].item()
            reward = compute_reward(action, true_label, t, episode_length)

            ep_reward += reward
            if action == true_label:
                correct_count += 1

            # Done flag: only at end of episode
            done = 1.0 if t == episode_length - 1 else 0.0

            buffer.add(obs_reconstructed.cpu(), action, log_prob.cpu(),
                       reward, value.cpu(), done)

            # Update prev action/reward for next step
            prev_action_oh = np.zeros(n_way, dtype=np.float32)
            prev_action_oh[action] = 1.0
            prev_reward_val = reward

        ep_accuracy = correct_count / episode_length
        return buffer, ep_reward, ep_accuracy

    # ─────────────────────────────────────────────
    # PPO UPDATE
    # ─────────────────────────────────────────────
    def ppo_update(self, buffer):
        """
        Runs PPO update on the collected episode buffer.

        Returns dict of loss components.
        """
        advantages, returns = buffer.compute_gae(self.gamma, self.gae_lambda)
        obs, actions, old_log_probs = buffer.get_tensors(self.device)

        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / \
                       (advantages_t.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        T = obs.shape[0]

        for _ in range(self.ppo_epochs):
            # Forward pass through the full sequence (preserve hidden state dynamics)
            obs_seq = obs.unsqueeze(0)  # (1, T, obs_dim)
            hidden = self.agent.init_hidden(batch_size=1, device=self.device)

            logits, values, _ = self.agent(obs_seq, hidden)
            # logits: (1, T, n_way), values: (1, T, 1)
            logits = logits.squeeze(0)   # (T, n_way)
            values = values.squeeze()    # (T,)

            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)  # (T,)
            entropy = dist.entropy().mean()          # scalar

            # Ratio: π_new / π_old
            ratio = torch.exp(new_log_probs - old_log_probs.detach())

            # Clipped surrogate objective
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                 1 + self.clip_eps) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss (clipped)
            value_loss = F.mse_loss(values, returns_t)

            # Total loss
            loss = (policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        n = self.ppo_epochs
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
        }

    # ─────────────────────────────────────────────
    # EVALUATION
    # ─────────────────────────────────────────────
    def evaluate(self, episode_gen, num_episodes=50):
        """
        Evaluate agent on held-out rare disease episodes.
        Returns mean reward and accuracy.
        """
        self.agent.eval()
        rewards, accs = [], []

        for _ in range(num_episodes):
            buffer, ep_reward, ep_acc = self.collect_episode(episode_gen)
            rewards.append(ep_reward)
            accs.append(ep_acc)

        self.agent.train()
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_accuracy": np.mean(accs) * 100,
            "std_accuracy": np.std(accs) * 100,
        }

    # ─────────────────────────────────────────────
    # FULL TRAINING LOOP
    # ─────────────────────────────────────────────
    def train(self, train_gen, val_gen,
              num_episodes=1000,
              eval_interval=50,
              save_path="checkpoints/rl2_best.pt"):
        """
        Full RL² training loop.

        Parameters
        ----------
        train_gen     : RL2EpisodeGenerator for meta-training diseases
        val_gen       : RL2EpisodeGenerator for rare disease testing
        num_episodes  : total training episodes
        eval_interval : evaluate every N episodes
        save_path     : best model checkpoint path
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        best_val_acc = 0.0

        print("\n" + "=" * 70)
        print("  RL² META-REINFORCEMENT LEARNING TRAINING")
        print(f"  Episodes: {num_episodes} | PPO epochs/ep: {self.ppo_epochs}")
        print(f"  γ: {self.gamma} | λ: {self.gae_lambda} | ε_clip: {self.clip_eps}")
        print("=" * 70)

        pbar = tqdm(range(1, num_episodes + 1), desc="RL² Training")
        recent_rewards = []
        recent_accs = []

        for episode_num in pbar:
            self.agent.train()

            # Collect episode
            buffer, ep_reward, ep_acc = self.collect_episode(train_gen)

            # PPO update
            loss_info = self.ppo_update(buffer)

            recent_rewards.append(ep_reward)
            recent_accs.append(ep_acc)

            # Keep rolling window
            if len(recent_rewards) > 20:
                recent_rewards.pop(0)
                recent_accs.pop(0)

            self.train_history["episode_reward"].append(ep_reward)
            self.train_history["episode_accuracy"].append(ep_acc * 100)
            self.train_history["policy_loss"].append(loss_info["policy_loss"])
            self.train_history["value_loss"].append(loss_info["value_loss"])
            self.train_history["entropy"].append(loss_info["entropy"])

            pbar.set_postfix({
                "rew": f"{np.mean(recent_rewards):.2f}",
                "acc": f"{np.mean(recent_accs)*100:.1f}%",
                "ploss": f"{loss_info['policy_loss']:.4f}",
            })

            # Periodic evaluation
            if episode_num % eval_interval == 0:
                val_results = self.evaluate(val_gen, num_episodes=30)

                self.val_history["episode"].append(episode_num)
                self.val_history["mean_reward"].append(val_results["mean_reward"])
                self.val_history["mean_accuracy"].append(val_results["mean_accuracy"])

                print(f"\n[Ep {episode_num:5d}] "
                      f"Train Reward: {np.mean(recent_rewards):.3f} | "
                      f"Train Acc: {np.mean(recent_accs)*100:.1f}% | "
                      f"Val Reward: {val_results['mean_reward']:.3f} | "
                      f"Val Acc: {val_results['mean_accuracy']:.1f}% ± "
                      f"{val_results['std_accuracy']:.1f}%")

                if val_results["mean_accuracy"] > best_val_acc:
                    best_val_acc = val_results["mean_accuracy"]
                    torch.save({
                        "episode": episode_num,
                        "model_state": self.agent.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "val_acc": best_val_acc,
                    }, save_path)
                    print(f"  ★ New best RL² model saved (acc: {best_val_acc:.1f}%)")

            self.scheduler.step()

        print(f"\n[RL²] Training complete. Best val accuracy: {best_val_acc:.2f}%")
        return self.train_history, self.val_history
