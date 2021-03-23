from carla_env import CarlaGym
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines import PPO2, A2C


NAME = "Changed Orientation"
LEARNING_RATE = 0.0003
GAMMA = 0.99
TRAINING_TIME = 1_000_000_000
SAVE_EVERY = 20_000

# Create a vectorized environment
env = CarlaGym()
check_env(env, warn=True)

env = DummyVecEnv([lambda: env])
env = VecCheckNan(env, raise_exception=True)

checkpoint_callback = CheckpointCallback(save_freq=SAVE_EVERY, save_path='./models/',
                                         name_prefix=NAME)

model = A2C("MlpPolicy", env, gamma=GAMMA, learning_rate=LEARNING_RATE, verbose=1,
            tensorboard_log="./logs/")

model.load("models/Steve_140000_steps.zip")

# Train the agent
model.learn(total_timesteps=TRAINING_TIME, tb_log_name=NAME,
            reset_num_timesteps=False, callback=checkpoint_callback)
