import retro
import gym
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from gym.wrappers import TimeLimit  

class CustomDiscretizer(gym.ActionWrapper):
    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()

class AirstrikerDiscretizer(CustomDiscretizer):
    def __init__(self, env):
        super().__init__(env=env, combos=[['RIGHT'],['LEFT'],['RIGHT', 'B'],['B'],['LEFT', 'B']])

def main():
    steps = 0
    env = retro.make(game='Airstriker-Genesis', use_restricted_actions=retro.Actions.DISCRETE)
    print(env.buttons)
    print('retro.Actions.DISCRETE action_space', env.action_space)
    env.close()

    env = retro.make(game='Airstriker-Genesis')
    env = AirstrikerDiscretizer(env)
    print('AirstrikerDiscretizer action_space', env.action_space)
    #env = MaxAndSkipEnv(env, 2)
    #env=TimeLimit(env,10000)
    obs = env.reset()
    print(obs.shape)
    done = False
    while not done:
        obs, rewards, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
        steps += 1
        if steps % 1000 == 0:
            print(f"Total Steps: {steps}")
            print(info)

    print("Final Info")
    print(info)
    env.close()

if __name__ == "__main__":
    main()
