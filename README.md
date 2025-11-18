# Chess-AlphaGO: GPU-Accelerated AlphaZero-Style RL with Pgx

**Paper:** *Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning* (NeurIPS 2023)  
**Pgx Codebase:** <https://github.com/sotetsuk/pgx>

This project explores **GPU-accelerated reinforcement learning** using the Pgx framework for board games.  
We focus on **Gardner chess** (a 5×5 mini-chess variant) and use the **official Pgx AlphaZero-style trainer** as our baseline, with small modifications to:

- Run on **GPU** with JAX.
- Train an **AlphaZero-style agent with MCTS** (Gumbel MuZero policy).
- Log per-iteration timing and throughput to a CSV file.
- Render the **final Gardner chess board** as an SVG image.

The long-term goal is to compare **CPU vs GPU** performance and scaling behavior (batch size, simulations, steps/sec).

---

## Background: Pgx and AlphaZero-Style Training

Pgx provides **GPU-friendly board game environments** written in JAX, including Go, Chess, Shogi, and Gardner chess.  
It supports batched, JIT-compiled rollouts, making it a natural fit for AlphaZero-style self-play with Monte Carlo Tree Search (MCTS).

In this repository we:

1. Use `pgx.make("gardner_chess")` to create a compact but non-trivial chess environment.
2. Use the **PGx AlphaZero example network** (`AZNet` in `network.py`) implemented with **Haiku**.
3. Run the original PGX **Gumbel MuZero / AlphaZero training loop** (`train.py`) with:
   - Batched self-play on GPU.
   - MCTS via `mctx.gumbel_muzero_policy`.
   - Periodic evaluation vs the PGX baseline model.
4. Add:
   - **Timing CSV logs** per iteration.
   - **Final SVG rendering** of a completed Gardner chess game.

---

## Repository Structure

```text
Chess-AlphaGO/
  README.md
  environment-pgx.yml        # Conda environment (JAX + Pgx + Haiku + Optax + etc.)

  src/
    __init__.py              # (optional) marks this as a package
    train.py                 # Main PGX AlphaZero-style trainer (modified)
    network.py               # AZNet definition used by train.py

  # Output artifacts (created at runtime; not necessarily committed)
  src/checkpoints/
    gardner_chess_YYYYMMDDHHMMSS/
      000000.ckpt            # Periodic model checkpoints (pickled dicts)
      000005.ckpt
      ...
      timing.csv             # Per-iteration timing / loss / throughput
      gardner_chess_final.svg# Final board rendering for a single game
