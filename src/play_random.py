# src/play_random.py

import jax
import jax.numpy as jnp
import pgx


def random_game(env_name: str = "gardner_chess", seed: int = 0):
    """
    Plays one random game in the given PGX environment and prints out rewards.
    Works for: "chess", "gardner_chess", etc.
    """
    print(f"Using devices: {jax.devices()}")
    print(f"Environment: {env_name}")

    key = jax.random.PRNGKey(seed)
    env = pgx.make(env_name)

    state = env.init(key)
    step_count = 0

    while not bool(state.terminated | state.truncated):
        mask = state.legal_action_mask  # shape [num_actions]

        # Give 0 logit to legal actions, -inf to illegal ones
        logits = jnp.where(mask, 0.0, -jnp.inf)

        key, subkey = jax.random.split(key)
        action = int(jax.random.categorical(subkey, logits))

        state = env.step(state, action)
        step_count += 1

        if step_count % 20 == 0:
            print(f"Step {step_count}, current_player={int(state.current_player)}, rewards={state.rewards}")

    print(f"Game finished in {step_count} steps")
    print("Final rewards:", state.rewards)


if __name__ == "__main__":
    # Try Gardner (small chess) first because it’s lighter
    random_game("gardner_chess", seed=0)
    # Uncomment to also test full chess:
    # random_game("chess", seed=1)
