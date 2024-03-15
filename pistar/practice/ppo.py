import sys  
sys.path.append("./src")

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from stable_baselines3 import PPO
    
import pandas as pd
import numpy as np
from gym_trading_env.environments import TradingEnv
import gymnasium as gym
from gym_trading_env.renderer import Renderer
import torch as th


df = pd.read_csv(r"C:\Users\user\Documents\GitHub\Gym-Trading-Env\pistar\data\SS00001.csv", parse_dates=["date"], index_col= "date")
df.dropna(inplace=True)

df['feature_close'] = (df['close'] - df['close'].min()) / (df['close'].max() - df['close'].min())
df["feature_volume"] = df['volume'].copy()
df["feature_RSI"] = df["RSI"].copy()
df["feature_MACD"] = df["MACD"].copy()
df["feature_CCI"] = df["CCI"].copy()
df["feature_ADX"] = df["ADX"].copy()

train_df = df.iloc[:-800] 
test_df = df.iloc[-800:]

# Create your own reward function with the history object
def log_return_reward_function(history):
    return 800*np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

def return_reward_function(history):
    return 1000*(history["portfolio_valuation", -1] - history["portfolio_valuation", -2]) / history["portfolio_valuation", -2]

def paper_reward_function(history):
    return (history["portfolio_valuation", -1] - history["portfolio_valuation", -2])

# Train Env 구성
train_env = gym.make(
        "TradingEnv",
        name= "stock",
        df = train_df,
        windows= 1,
        positions = [0, 1, 2], # From -1 (=SHORT), to +1 (=LONG)
        initial_position = 0, # Initial position
        trading_fees = 0.01/100, # 0.01% per stock buy / sell
        fiat_borrow_interest_rate = 0,
        asset_borrow_interest_rate= 0,
        reward_function = log_return_reward_function,
        portfolio_initial_value = 1000, # in FIAT (here, USD)
        max_episode_duration = 'max',
    )

# Train Env에서 metric을 추가해줍니다.
train_env.unwrapped.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
train_env.unwrapped.add_metric('Episode Lenght', lambda history : len(history['position']) )

# Test Env 구성
test_env = gym.make(
        "TradingEnv",
        name= "stock",
        df = test_df,
        windows= 1,
        positions = [0, 1, 2], # From -1 (=SHOR), to +1 (=LONG)
        initial_position = 0, # Initial position
        trading_fees = 0.01/100, # 0.01% per stock buy / sell
        fiat_borrow_interest_rate = 0,
        asset_borrow_interest_rate= 0,
        reward_function = log_return_reward_function,
        portfolio_initial_value = 1000, # in FIAT (here, USD)
        max_episode_duration = 'max',
    )

# Test Env에 metric을 추가해줍니다.
test_env.unwrapped.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0))
test_env.unwrapped.add_metric('Episode Lenght', lambda history : len(history['position']))


# 논문에 나온대로 신경망의 크기를 조절해줍니다.
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[512, 256], vf=[512, 256]))

# 모델을 불러옵니다.
model = PPO(policy="MlpPolicy",
            tensorboard_log="C:/Users/user/Documents/GitHub/Gym-Trading-Env//pistar/log/pistar_tensorboard/",
            env=train_env, 
            n_steps=128,
            vf_coef = 0.1,
            clip_range=0.05,
            ent_coef=0.001, 
            policy_kwargs=policy_kwargs,
            verbose=0)


print("=========================train=========================")
model.learn(total_timesteps=25_000, tb_log_name="ppo")

# Save for render
train_env.unwrapped.save_for_render(log_name = "ppo_train")


print("=========================test=========================")

# 학습을 하진 않고 predict만 진행하게 됩니다. 

done, truncated = False, False
observation, info = test_env.reset()

while not done and not truncated:
    action, _states = model.predict(observation)
    observation, reward, done, truncated, info = test_env.step(action)
    print(observation)
    
# Save for render
test_env.unwrapped.save_for_render(log_name="ppo_test")

    
renderer = Renderer(render_logs_dir="render_logs")
renderer.run()