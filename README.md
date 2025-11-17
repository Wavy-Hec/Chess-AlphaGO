# Chess-AlphaGO: GPU-Accelerated AlphaZero-Style RL with Pgx

**Paper:** *Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning* (NeurIPS 2023)  
**Pgx Codebase:** <https://github.com/sotetsuk/pgx>  

This project explores **GPU-accelerated reinforcement learning** using the Pgx framework.  
We focus on an **AlphaZero-style self-play agent** for **Gardner chess** (a 5x5 mini-chess variant) and later extend to full chess.  

Our main goals are:

- Implement a **GPU-native** RL pipeline using JAX + Pgx.
- Compare **CPU vs GPU** training performance (environment steps/sec and wall-clock time).
- Log how training scales with **batch size**, **replay buffer size**, and **total environment steps**.
- Produce visual outputs (board renderings and graphs) for analysis.

---

## Background: Pgx and GPU-Native RL

Pgx provides a set of **GPU-friendly board-game simulators** written in JAX, including Go, Chess, Shogi, and Gardner chess.  
It supports batched, JIT-compiled rollouts entirely on GPU, enabling high-throughput self-play and making it a good testbed for AlphaZero-style algorithms.

In this repository we:

1. Use `pgx.make("gardner_chess")` to create a small but non-trivial chess environment.
2. Implement an **AlphaZero-style agent** with:
   - Convolutional torso with **policy + value heads** (Flax).
   - **Self-play** where the network plays both sides (no MCTS yet; policy is sampled directly).
   - A **replay buffer** storing (observation, policy target, value target).
3. Train the agent on **GPU** and log detailed timing information.
4. Render the final game board as an **SVG image** for visualization.

Later phases will add:

- Full chess environment (`pgx.make("chess")`).
- Search (MCTS / Gumbel AlphaZero) on top of the policy/value network.
- More systematic CPU vs GPU experiments.

---

## Repository Structure

```text
Chess-AlphaGO/
  README.md
  src/
    __init__.py
    play_random.py         # Simple random self-play for sanity checking PGX
    train_policy_gpu.py    # Supervised policy-only toy training (legal-move predictor)
    alpha_zero_gardner.py  # Main AlphaZero-style self-play + training loop
  checkpoints/             # Saved model parameters (.msgpack)
  logs/                    # Timing CSV for throughput / scaling analysis
  renders/                 # SVG board renderings of final games
