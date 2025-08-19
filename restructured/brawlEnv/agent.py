from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from brawlEnv import BrawlStarsEnv

#Brawlstars environment
env = DummyVecEnv([lambda: BrawlStarsEnv()])

agent= PPO(
    policy="CnnPolicy",
    env=env,
    learning_rate=0.0001,
    n_steps=128,
    batch_size=64,
    gamma=0.99,
    n_epochs=10,
    ent_coef=0.01,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./ppo_brawlstars_tensorboard/",
)  

# ppo_model.learn(total_timesteps=1000000, tb_log_name="ppo_brawlstars")
