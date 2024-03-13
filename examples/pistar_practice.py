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


df = pd.read_csv(r"C:\Users\user\Documents\GitHub\Gym-Trading-Env\examples\data\pistar\SS00001.csv", parse_dates=["date"], index_col= "date")
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

def paper_reward_function(history):
    return (history["portfolio_valuation", -1] - history["portfolio_valuation", -2]) / history["portfolio_valuation", -2]


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

train_env.unwrapped.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
train_env.unwrapped.add_metric('Episode Lenght', lambda history : len(history['position']) )

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

test_env.unwrapped.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0))
test_env.unwrapped.add_metric('Episode Lenght', lambda history : len(history['position']))

observation, info = train_env.reset()
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[512, 256], vf=[512, 256]))

model = PPO(policy="MlpPolicy", 
            env=train_env, 
            n_steps=128,
            clip_range=0.1,
            ent_coef=0.005, 
            policy_kwargs=policy_kwargs)


print("=========================train=========================")
model.learn(total_timesteps=100_000)


print("=========================test=========================")

done, truncated = False, False
observation, info = test_env.reset()

while not done and not truncated:
    action, _states = model.predict(observation)
    observation, reward, done, truncated, info = test_env.step(action)
    
# Save for render
test_env.unwrapped.save_for_render()

    
renderer = Renderer(render_logs_dir="render_logs")
renderer.run()