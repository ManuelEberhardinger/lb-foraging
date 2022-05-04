import argparse
from fileinput import close
import logging
import random
import time
import gym
import numpy as np
import lbforaging

from lbforaging.foraging.environment import Action

logger = logging.getLogger(__name__)


def get_accessible_obs(obs):
    field_size = obs.shape[-1]
    o = np.array(obs).reshape((field_size*3, field_size))
    agent = o[:field_size, :]
    food = o[field_size:field_size*2, :]
    access = o[field_size*2:, :]
    nonzeros = np.nonzero(access)
    x1, x2, y1, y2 = np.min(nonzeros[0]), np.max(nonzeros[0]) + 1, np.min(nonzeros[1]), np.max(nonzeros[1]) + 1
    return agent[x1:x2, y1:y2], food[x1:x2, y1:y2], access[x1:x2, y1:y2]


def closest_node(node, nodes, mask):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    dist = np.ma.MaskedArray(dist_2, mask)
    return np.ma.argmin(dist)


def get_new_agent_pos(pos, action):
    if action == action.NORTH:
        return (pos[0] - 1, pos[1])
    elif action == action.SOUTH:
        return (pos[0] + 1, pos[1])
    elif action == action.WEST:
        return (pos[0], pos[1] - 1)
    elif action == action.EAST:
        return (pos[0], pos[1] + 1)
    else:
        return pos


def _expert_policy(obs, info):
    pos = info['player_pos']
    actions = []
    new_pos = []
    collab_food_pos = None
    for i in range(len(obs)):
        agent, food, access = get_accessible_obs(obs[i])
        a_pos = pos[i]
        agent_lvl = agent[a_pos[0], a_pos[1]]
        # print(agent_lvl)
        # first find food in observation
        food_pos = np.argwhere(food > 0)

        food_lvls = [food[p[0], p[1]] for p in food_pos]
        can_not_pickup = [fl > agent_lvl for fl in food_lvls]
        if all(can_not_pickup) and collab_food_pos is not None:
            nearest_food = collab_food_pos
        elif all(can_not_pickup):
            can_not_pickup = [False for _ in food_lvls]
            nearest_food = food_pos[closest_node(a_pos, food_pos, can_not_pickup)]
        else:
            nearest_food = food_pos[closest_node(a_pos, food_pos, can_not_pickup)]

        if collab_food_pos is None:
            collab_food_pos = nearest_food

        if abs(a_pos[0] - nearest_food[0]) > 1:
            if a_pos[0] - nearest_food[0] > 0 and (a_pos[0] - 1, a_pos[1]) not in new_pos and access[a_pos[0] - 1, a_pos[1]]:
                actions.append(Action.NORTH)
            elif (a_pos[0] + 1, a_pos[1]) not in new_pos and access[a_pos[0] + 1, a_pos[1]]:
                actions.append(Action.SOUTH)
            else:
                actions.append(Action.NONE)
        elif abs(a_pos[1] - nearest_food[1]) > 1:
            if a_pos[1] - nearest_food[1] < 0 and (a_pos[0], a_pos[1] + 1) not in new_pos and access[a_pos[0], a_pos[1] + 1]:
                actions.append(Action.EAST)
            elif (a_pos[0], a_pos[1] - 1) not in new_pos and access[a_pos[0], a_pos[1] - 1]:
                actions.append(Action.WEST)
            else:
                actions.append(Action.NONE)
        elif abs(a_pos[0] - nearest_food[0]) == 1 and abs(a_pos[1] - nearest_food[1]) == 1:
            if a_pos[0] - nearest_food[0] > 0 and access[a_pos[0] - 1, a_pos[1]] and (a_pos[0] - 1, a_pos[1]) not in new_pos:
                actions.append(Action.NORTH)
            elif a_pos[0] - nearest_food[0] < 0 and access[a_pos[0] + 1, a_pos[1]] and (a_pos[0] + 1, a_pos[1]) not in new_pos:
                actions.append(Action.SOUTH)
            elif a_pos[1] - nearest_food[1] > 0 and access[a_pos[0], a_pos[1] - 1] and (a_pos[0], a_pos[1] - 1) not in new_pos:
                actions.append(Action.WEST)
            elif a_pos[1] - nearest_food[1] < 0 and access[a_pos[0], a_pos[1] + 1] and (a_pos[0], a_pos[1] + 1) not in new_pos:
                actions.append(Action.EAST)
            else:
                actions.append(Action.NONE)
        else:
            actions.append(Action.LOAD)

        new_pos.append(get_new_agent_pos(a_pos, actions[-1]))
    print(actions)
    return tuple(actions)


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
        actions = _expert_policy(obs, info)

        obs, nreward, ndone, info = env.step(actions)

        if sum(nreward) > 0:
            print(nreward)

        if render:
            env.render()
            time.sleep(0.5)

        done = np.all(ndone)
    print(env.players[0].score, env.players[1].score)


def main(game_count=1, render=False):
    env = gym.make("Foraging-grid-10x10-2p-4f-coop-v2")
    obs = env.reset()

    for episode in range(game_count):
        _game_loop(env, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    args = parser.parse_args()
    main(args.times, args.render)
