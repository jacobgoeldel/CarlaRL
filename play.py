from carla_env import CarlaGym
from stable_baselines.common.env_checker import check_env
from stable_baselines import PPO2, A2C
import time

env = CarlaGym()

check_env(env, warn=True)

model = PPO2("CnnPolicy", env, verbose=0)

model.load("models/Charles.zip")

obs = env.reset()
for i in range(5000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    time.sleep(0.03)  # hacky way to slow down simulation to around real time
    if done:
        obs = env.reset()
env.close()
