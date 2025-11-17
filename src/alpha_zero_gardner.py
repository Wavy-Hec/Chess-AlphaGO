import os
import time
import csv
import random
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
import pgx
from flax import linen as nn
from flax.training import train_state
from flax import serialization
from tqdm import trange


# =========================
# Config
# =========================

@dataclass
class Config:
    env_name: str = "gardner_chess"
    seed: int = 0

    # Self-play
    num_games: int = 100          # how many self-play games to generate
    max_moves_per_game: int = 128 # safety cap

    # Replay buffer
    replay_capacity: int = 20_000

    # Training
    batch_size: int = 256
    lr: float = 3e-4
    updates_per_game: int = 4     # gradient steps after each game

    # Logging / checkpoints
    timing_csv: str = "logs/timing_gardner.csv"
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 20    # games
    render_dir: str = "renders"
    render_final_svg: str = "renders/final_gardner_game.svg"


CFG = Config()


# =========================
# Network: policy + value head
# =========================

class AZNet(nn.Module):
    num_actions: int
    channels: int = 64

    @nn.compact
    def __call__(self, x):
        """
        x: [B, H, W, C] observation
        Returns:
          policy_logits: [B, num_actions]
          value: [B] in [-1, 1]
        """
        x = x.astype(jnp.float32)
        # Simple conv torso
        x = nn.Conv(self.channels, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(nn.Dense(128)(x))

        policy_logits = nn.Dense(self.num_actions)(x)
        value = nn.tanh(nn.Dense(1)(x))  # scalar in [-1, 1]
        return policy_logits, value[..., 0]


class TrainState(train_state.TrainState):
    # we’ll keep apply_fn the same: params -> (policy_logits, value)
    pass


def create_train_state(rng, obs_example, num_actions, lr):
    model = AZNet(num_actions=num_actions)
    params = model.init(rng, obs_example[None, ...])  # add batch dim
    tx = optax.adam(lr)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# =========================
# Replay buffer
# =========================

class ReplayBuffer:
    """
    Stores (obs, pi, z).
      obs: [H, W, C]
      pi:  [num_actions] (prob dist)
      z:   scalar in [-1, 1] (outcome from player’s perspective)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = []

    def add_many(self, samples):
        for s in samples:
            self.add(*s)

    def add(self, obs, pi, z):
        if len(self.data) >= self.capacity:
            # Drop oldest
            self.data.pop(0)
        self.data.append((obs, pi, z))

    def __len__(self):
        return len(self.data)

    def sample_batch(self, batch_size: int, rng: random.Random):
        idxs = rng.sample(range(len(self.data)), batch_size)
        obs_batch = []
        pi_batch = []
        z_batch = []
        for i in idxs:
            obs, pi, z = self.data[i]
            obs_batch.append(obs)
            pi_batch.append(pi)
            z_batch.append(z)
        return (
            jnp.stack(obs_batch),           # [B, H, W, C]
            jnp.stack(pi_batch),            # [B, num_actions]
            jnp.array(z_batch, dtype=jnp.float32),  # [B]
        )


# =========================
# Self-play (simplified AlphaZero-style, no MCTS)
# =========================

def softmax_masked(logits, mask, temperature: float = 1.0):
    """Masked softmax with temperature. Illegal moves get prob 0."""
    # Set very negative logits for illegal moves
    masked_logits = jnp.where(mask, logits, -jnp.inf)
    if temperature != 1.0:
        masked_logits = masked_logits / temperature
    probs = jax.nn.softmax(masked_logits)
    # Replace NaNs (e.g., all -inf) with uniform over legal moves
    probs = jnp.nan_to_num(probs, nan=0.0)
    legal = mask.astype(jnp.float32)
    legal_sum = legal.sum()
    probs = jnp.where(legal_sum > 0, probs, legal / jnp.maximum(legal_sum, 1.0))
    return probs


def play_one_game(env, params, apply_fn, rng_key, max_moves: int):
    """
    Self-play one Gardner chess game using the current network policy.
    Returns:
      samples: list of (obs, pi, z)
      rng_key: updated key
      final_state: last PGX state (for rendering)
      num_moves: number of moves taken
    """
    # Initialize state
    rng_key, sub = jax.random.split(rng_key)
    state = env.init(sub)

    history = []  # list of (obs, player_id, pi)
    num_moves = 0

    while not bool(state.terminated | state.truncated) and num_moves < max_moves:
        obs = state.observation
        mask = state.legal_action_mask
        current_player = int(state.current_player)

        # Network forward
        logits, value = apply_fn(params, obs[None, ...])
        logits = logits[0]

        # Masked policy with temperature
        pi = softmax_masked(logits, mask, temperature=1.0)

        # Sample action from pi
        rng_key, sub = jax.random.split(rng_key)
        action = int(jax.random.categorical(sub, jnp.log(pi + 1e-20)))

        history.append((obs, current_player, pi))

        state = env.step(state, action)
        num_moves += 1

    # Final outcome: rewards per player
    # Gardner chess has 2 players; rewards in {-1,0,1}.:contentReference[oaicite:1]{index=1}
    rewards = state.rewards

    samples = []
    for obs, player_id, pi in history:
        z = float(rewards[player_id])  # perspective of the player who moved
        samples.append((obs, pi, z))

    return samples, rng_key, state, num_moves


# =========================
# Loss and train step
# =========================

def loss_fn(params, apply_fn, obs_batch, pi_batch, z_batch):
    """
    Standard AlphaZero-style loss:
      L = (z - v)^2 - pi^T log p  (up to constants)
    """
    logits, values = apply_fn(params, obs_batch)  # logits: [B, A], values: [B]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # Policy loss: cross-entropy
    policy_loss = -(pi_batch * log_probs).sum(axis=-1).mean()
    # Value loss: MSE
    value_loss = jnp.mean((z_batch - values) ** 2)
    # No explicit regularization for now
    total_loss = policy_loss + value_loss
    return total_loss, (policy_loss, value_loss)

@jax.jit
def train_step(state: TrainState, obs_batch, pi_batch, z_batch):
    def _loss_fn(params):
        total, (pol, val) = loss_fn(
            params,
            state.apply_fn,
            obs_batch,
            pi_batch,
            z_batch,
        )
        return total, (pol, val)

    # Compute loss and gradients in one go
    (total, (pol, val)), grads = jax.value_and_grad(
        _loss_fn,
        has_aux=True,
    )(state.params)

    # Gradient update
    state = state.apply_gradients(grads=grads)

    # Return updated state + losses (so you can log them later)
    return state, total, pol, val



# =========================
# Checkpoint helpers
# =========================

def save_checkpoint(state: TrainState, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    params_cpu = jax.device_get(state.params)
    bytes_out = serialization.to_bytes(params_cpu)
    with open(path, "wb") as f:
        f.write(bytes_out)
    print(f"[checkpoint] Saved params to {path}")


# =========================
# Main training loop
# =========================

def train_alpha_zero_gardner():
    print("JAX devices:", jax.devices())
    print("Config:", CFG)

    # Make env & get action space / obs shape
    env = pgx.make(CFG.env_name)
    key = jax.random.PRNGKey(CFG.seed)
    key, sub = jax.random.split(key)
    state0 = env.init(sub)
    obs0 = state0.observation
    num_actions = int(state0.legal_action_mask.shape[0])

    print("Observation shape:", obs0.shape)
    print("Num actions:", num_actions)

    # Init model / optimizer
    key, sub = jax.random.split(key)
    train_state_obj = create_train_state(sub, obs0, num_actions, CFG.lr)

    # Replay buffer + Python RNG for sampling
    buffer = ReplayBuffer(CFG.replay_capacity)
    py_rng = random.Random(CFG.seed)

    # Timing CSV
    os.makedirs(os.path.dirname(CFG.timing_csv), exist_ok=True)
    timing_file = open(CFG.timing_csv, "w", newline="")
    timing_writer = csv.writer(timing_file)
    timing_writer.writerow(
        ["global_step", "game_idx", "total_env_steps",
         "buffer_size", "batch_size", "replay_capacity",
         "elapsed_sec", "steps_per_sec"]
    )

    total_env_steps = 0
    global_step = 0
    train_start_time = time.perf_counter()

    last_state_for_render = state0

    for game_idx in trange(CFG.num_games, desc="Self-play + train"):
        # --- Self-play game ---
        samples, key, last_state, num_moves = play_one_game(
            env,
            train_state_obj.params,
            train_state_obj.apply_fn,
            key,
            max_moves=CFG.max_moves_per_game,
        )
        last_state_for_render = last_state
        total_env_steps += num_moves

        # Add to buffer
        buffer.add_many(samples)

        # --- Training updates ---
        num_updates = min(CFG.updates_per_game, len(buffer) // CFG.batch_size)
        for _ in range(num_updates):
            obs_b, pi_b, z_b = buffer.sample_batch(CFG.batch_size, py_rng)
            train_state_obj, total_loss, pol_loss, val_loss = train_step(
                train_state_obj, obs_b, pi_b, z_b
            )
            global_step += 1

            elapsed = time.perf_counter() - train_start_time
            steps_per_sec = total_env_steps / max(elapsed, 1e-6)
            timing_writer.writerow(
                [
                    global_step,
                    game_idx,
                    total_env_steps,
                    len(buffer),
                    CFG.batch_size,
                    CFG.replay_capacity,
                    elapsed,
                    steps_per_sec,
                ]
            )

        if game_idx % 10 == 0:
            print(
                f"[game {game_idx}] moves={num_moves}, "
                f"buffer={len(buffer)}, total_env_steps={total_env_steps}"
            )

        # Periodic checkpoints
        if (game_idx + 1) % CFG.checkpoint_every == 0:
            ckpt_path = os.path.join(
                CFG.checkpoint_dir,
                f"az_gardner_game{game_idx+1}.msgpack",
            )
            save_checkpoint(train_state_obj, ckpt_path)

    timing_file.close()

    # Final checkpoint
    final_ckpt = os.path.join(CFG.checkpoint_dir, "az_gardner_final.msgpack")
    save_checkpoint(train_state_obj, final_ckpt)

    # Render final board as SVG
    os.makedirs(CFG.render_dir, exist_ok=True)
    last_state_for_render.save_svg(CFG.render_final_svg, color_theme="dark", scale=2.0)
    print(f"[render] Saved final board to {CFG.render_final_svg}")


if __name__ == "__main__":
    train_alpha_zero_gardner()
