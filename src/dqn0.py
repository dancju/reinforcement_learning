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
            assert i_obs == self.model.layers[0].input_shape[1]
            assert i_act == self.model.layers[-1].output_shape[1]
        elif type(model) == tuple:
            self.model = Sequential()
            self.model.add(Input((i_obs,)))
            for i in model:
                self.model.add(Dense(i, activation="relu"))
            self.model.add(Dense(i_act, activation="linear"))
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
        return np.argmax(self.model.predict(ar_obs.reshape(1, self.i_obs)))

    def play(self, i_batch=128):
        if len(self.memory) < i_batch:
            return 0
        ar_batch = sample(self.memory, i_batch)

        ar_x = np.array([i[0] for i in ar_batch])
        ar_y = self.model.predict(ar_x)
        ar_q = self.model.predict(np.array([i[2] for i in ar_batch])).max(axis=1)
        for i in range(i_batch):
            _, i_act, _, f_reward, b_done = ar_batch[i]
            ar_y[i, i_act] = f_reward if b_done else f_reward + self.f_gamma * ar_q[i]
        return self.model.fit(ar_x, ar_y, i_batch, verbose=0).history["loss"][0]

        # for ar_obs, i_act, ar_obs_next, f_reward, b_done in ar_batch:
        #     f_q = f_reward
        #     if not b_done:
        #         f_q += self.f_gamma * np.max(
        #             self.model.predict(ar_obs_next.reshape(1, self.i_obs))
        #         )
        #     ar_y = self.model.predict(ar_obs.reshape(1, self.i_obs))
        #     ar_y[0, i_act] = f_q
        #     ar_x = ar_obs.reshape(1, self.i_obs)
        #     self.model.fit(ar_x, ar_y, verbose=0)
        # return 0

    def save(self, str_file):
        self.model.save(str_file)
