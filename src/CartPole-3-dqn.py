from random import random

import gym

from dqn0 import DQN

F_EXPLORATION_DECAY = 0.9999
F_EXPLORATION_BOUND = 0.0001
STR_MODEL = "src/CartPole-3-dqn.tf"


def get_action_(f_y, f_x):
    return f_y - f_x ** 2 / 20 * (-1 if f_x > 0 else 1)


def get_action(ar_obs, f_threshold=1e-2):
    f_pos, f_vol_pos, f_ang, f_vol_ang = ar_obs
    f_act_pos = get_action_(f_pos, f_vol_pos)
    f_act_ang = get_action_(f_ang, f_vol_ang)
    f_act = f_act_ang if abs(f_act_ang) > f_threshold else f_act_pos
    return int(f_act > 0)


def main():
    env = gym.make("CartPole-v1")
    i_epoch = 0
    try:
        solver = DQN(4, 2, STR_MODEL)
        f_exploration = F_EXPLORATION_BOUND
    except IOError:
        solver = DQN(4, 2, (16, 16))
        f_exploration = 1.0
    try:
        while True:
            ar_obs = env.reset()
            i_step = 0
            while True:
                i_act = (
                    get_action(ar_obs)
                    if random() < f_exploration
                    else solver.get_action(ar_obs)
                )
                f_exploration *= F_EXPLORATION_DECAY
                f_exploration = max(f_exploration, F_EXPLORATION_BOUND)
                ar_obs_next, f_reward, b_done, _ = env.step(i_act)
                solver.record(ar_obs, i_act, ar_obs_next, f_reward, b_done)
                ar_obs = ar_obs_next
                i_step += 1
                if b_done:
                    break
            print(
                f"{i_epoch:3} | "
                f"exploration: {f_exploration:6.4f} | "
                f"score: {i_step:3} | "
                f"loss: {solver.play():5.2f}"
            )
            i_epoch += 1
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Saving model...")
        solver.save(STR_MODEL)


if __name__ == "__main__":
    main()
