import gym
import numpy as np
import pandas as pd
from keras.layers import Dense, Input
from keras.models import Sequential


def get_act_(f_y, f_x):
    return f_y - f_x ** 2 / 20 * (-1 if f_x > 0 else 1)


def get_act(ar_obs, f_threshold=1e-2):
    f_pos, f_vol_pos, f_ang, f_vol_ang = ar_obs
    f_act_pos = get_act_(f_pos, f_vol_pos)
    f_act_ang = get_act_(f_ang, f_vol_ang)
    f_act = f_act_ang if abs(f_act_ang) > f_threshold else f_act_pos
    return int(f_act > 0)


def get_data(env):
    ar0 = []
    for i in range(8):
        ar1 = []
        ar_obs = env.reset()
        while True:
            i_act = get_act(ar_obs)
            ar1.append(ar_obs.tolist() + [int(i_act == 0), int(i_act == 1)])
            ar_obs, f_reward, b_done, _ = env.step(i_act)
            if b_done:
                break
        assert len(ar1) == 500
        ar0.extend(ar1)
    return pd.DataFrame(
        ar0, columns=["pos", "velo_pos", "ang", "velo_ang", "act_0", "act_1"]
    )


def get_model(df):
    model = Sequential()
    model.add(Input((4,)))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(2, activation="sigmoid"))
    model.compile("adam", "categorical_crossentropy")
    model.fit(
        df[["pos", "velo_pos", "ang", "velo_ang"]], df[["act_0", "act_1"]], epochs=32
    )
    return model


def test(env, model):
    for i_epoch in range(8):
        i_reward = 0
        ar_obs = env.reset()
        while True:
            ar_action = model.predict(ar_obs.reshape(1, 4))
            i_action = np.argmax(ar_action)
            ar_obs, _, b_done, _ = env.step(i_action)
            i_reward += 1
            if b_done:
                break
        print(f"Epoch finished after {i_reward} steps")


def main():
    env = gym.make("CartPole-v1")
    print("Stage 0: Generating raw data through a heuristic strategy.")
    df = get_data(env)
    print("Stage 1: Training.")
    model = get_model(df)
    print("Stage 2: Testing.")
    test(env, model)


if __name__ == "__main__":
    main()
