import gym


def get_action_(f_y, f_x, f_tau=0.02):
    return f_y + f_x * f_tau


def get_action(ar_obs, f_threshold=1e-2):
    f_pos, f_vol_pos, f_ang, f_vol_ang = ar_obs
    f_act_pos = get_action_(f_pos, f_vol_pos)
    f_act_ang = get_action_(f_ang, f_vol_ang)
    f_act = f_act_ang if abs(f_act_ang) > f_threshold else f_act_pos
    return int(f_act > 0)


def main():
    env = gym.make("CartPole-v1")
    ar_obs = env.reset()
    i = 0
    while True:
        env.render()
        ar_obs, f_reward, b_done, _ = env.step(get_action(ar_obs))
        i += 1


if __name__ == "__main__":
    main()
