import retro
from gym.wrappers import TimeLimit
from RandomAgent import AirstrikerDiscretizer
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from gym.wrappers import ResizeObservation, GrayScaleObservation
from gym.envs.classic_control import rendering
import numpy as np

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!").format(k, l)
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

model = PPO.load("tmp/best_model_airstriker.zip")
#model=PPO.load("SuperMarioBros-Nes.zip")
def main():
    steps = 0
    viewer = rendering.SimpleImageViewer(maxwidth=1600)
    env = retro.make(game='Airstriker-Genesis')
    env = AirstrikerDiscretizer(env)
    #env = TimeLimit(env,10000)
    #env = MaxAndSkipEnv(env, 2)
    env=ResizeObservation(env,84)
    env=GrayScaleObservation(env, keep_dim=True)
    
    obs = env.reset()
    done = False

    while not done:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        #env.render()
        rgb = env.render('rgb_array')
        upscaled=repeat_upsample(rgb,3, 3)
        viewer.imshow(upscaled)
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