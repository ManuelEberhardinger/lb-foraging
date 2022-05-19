import argparse
from fileinput import close
import logging
import random
import time
import gym
import numpy as np
from lbforaging.agents.expert_policy import expert_policy

logger = logging.getLogger(__name__)


def _game_loop(env, render):
    """
    """
    obs = env.reset()
    info = env.get_player_pos_info()

    done = False

    if render:
        env.render()
        time.sleep(0.5)

    while not done:
        actions = expert_policy(obs, info)

        obs, nreward, ndone, info = env.step(actions)

        if sum(nreward) > 0:
            print(nreward)

        if render:
            env.render()
            time.sleep(0.5)

        done = np.all(ndone)
    print(env.players[0].score, env.players[1].score)


def main(env_name, game_count=1, render=False):
    env = gym.make(env_name)
    obs = env.reset()

    for episode in range(game_count):
        _game_loop(env, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )
    parser.add_argument(
        "--env", type=str, default="Foraging-grid-10x10-5p-5f-v2", help="The env to run"
    )

    args = parser.parse_args()
    main(args.env, args.times, args.render)
