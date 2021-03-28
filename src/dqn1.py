from collections import deque
from random import sample

import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential, load_model
from keras.optimizers import Adam


class DQN:
    def __init__(self, i_obs, i_act, model, f_gamma=0.95, i_memory=65536):
        if type(i_obs) != int or i_obs <= 0:
            raise ValueError
        self.i_obs = i_obs
        if type(i_act) != int or i_act <= 0:
            raise ValueError
        self.i_act = i_act
        if type(model) == str:
            self.model = load_model(model)
            assert self.model.layers[0].input_shape[1] == i_obs + i_act
        elif type(model) == tuple:
            self.model = Sequential()
            self.model.add(Input((i_obs + i_act,)))
            for i in model:
                self.model.add(Dense(i, activation="relu"))
            self.model.add(Dense(1, activation="linear"))
            self.model.compile(Adam(), "mse")
        else:
            raise TypeError
        if type(f_gamma) != float or f_gamma <= 0 or f_gamma >= 1:
            raise ValueError
        self.f_gamma = f_gamma
        if type(i_memory) != int or i_memory <= 0:
            raise ValueError
        self.memory = deque(maxlen=i_memory)

    def record(self, ar_obs, i_act, ar_obs_next, f_reward, b_done):
        if type(ar_obs) != np.ndarray or ar_obs.shape != (self.i_obs,):
            raise ValueError
        if not isinstance(i_act, (int, np.integer)) or i_act < 0 or i_act >= self.i_act:
            raise ValueError
        if type(ar_obs_next) != np.ndarray or ar_obs_next.shape != (self.i_obs,):
            raise ValueError
        if type(f_reward) != float:
            raise TypeError
        if type(b_done) != bool:
            raise TypeError
        self.memory.append((ar_obs, i_act, ar_obs_next, f_reward, b_done))

    def get_action(self, ar_obs):
        if type(ar_obs) != np.ndarray or ar_obs.shape != (self.i_obs,):
            raise ValueError
        return np.argmax(
            [
                self.model.predict(
                    np.concatenate([ar_obs, np.arange(self.i_act) == i]).reshape(
                        1, self.i_obs + self.i_act
                    )
                )
                for i in range(self.i_act)
            ]
        )

    def play(self, i_batch=128):
        if len(self.memory) < i_batch:
            return 0
        ar_batch = sample(self.memory, i_batch)
        ar_x = np.array(
            [np.concatenate([i[0], np.arange(self.i_act) == i[1]]) for i in ar_batch]
        )
        assert ar_x.shape == (i_batch, 6)
        ar_y = [
            np.concatenate(
                [
                    np.repeat(i[2].reshape(1, self.i_obs), self.i_act, 0),
                    np.identity(self.i_act),
                ],
                1,
            )
            for i in ar_batch
        ]
        ar_y = np.array(ar_y)
        assert ar_y.shape == (i_batch, 2, 6)
        ar_y = self.model.predict(ar_y)
        ar_y = ar_y.reshape(i_batch, 2).max(axis=1)
        for i in range(i_batch):
            _, _, _, f_reward, b_done = ar_batch[i]
            ar_y[i] = f_reward if b_done else f_reward + ar_y[i]
        return self.model.fit(ar_x, ar_y, i_batch, verbose=0).history["loss"][0]

    def save(self, str_file):
        self.model.save(str_file)
