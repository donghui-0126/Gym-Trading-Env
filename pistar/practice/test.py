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

train_df = df.iloc[:-800].copy()
test_df = df.iloc[-800:].copy()

train_df['feature_close'] = (train_df['close'] - train_df['close'].min()) / (train_df['close'].max() - train_df['close'].min())
train_df["feature_volume"] = train_df['volume'].copy()
train_df["feature_RSI"] = train_df["RSI"].copy()
train_df["feature_MACD"] = train_df["MACD"].copy()
train_df["feature_CCI"] = train_df["CCI"].copy()
train_df["feature_ADX"] = train_df["ADX"].copy()

test_df['feature_close'] = (test_df['close'] - test_df['close'].min()) / (test_df['close'].max() - test_df['close'].min())
test_df["feature_volume"] = test_df['volume'].copy()
test_df["feature_RSI"] = test_df["RSI"].copy()
test_df["feature_MACD"] = test_df["MACD"].copy()
test_df["feature_CCI"] = test_df["CCI"].copy()
test_df["feature_ADX"] = test_df["ADX"].copy()

def basic_reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

def basic_reward_function_when_execute(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", 0])

def sharpe_tanh_reward_function_when_execute(history):
    sharpe_ratio = np.sum(history['portfolio_return', -1])/np.std(history['portfolio_return', -1])
    reward = np.tanh(sharpe_ratio)
    return reward
##################################################

def dynamic_feature_last_position_taken(history):
    return history['position', -1] / 10

def dynamic_feature_real_position(history):
    return history['real_position', -1] / 10

def dynamic_feature_asset(history):
    return history['asset', -1] / 1000

def dynamic_feature_fiat(history):
    return history['fiat', -1] / 1000

def dynamic_feature_step(history):
    return history['step', -1] / 128

# Test Env 구성
test_env = gym.make(
        "TradingEnv",
        name= "stock",
        df = test_df,
        positions = [-5,-3,-1,0,1,3,5],
        dynamic_feature_functions = [dynamic_feature_last_position_taken,  
                                    dynamic_feature_asset, 
                                    dynamic_feature_fiat,
                                    dynamic_feature_step],
        reward_function_when_execute = sharpe_tanh_reward_function_when_execute,
        windows = 30,
        trading_fees = 0.04/100,
        portfolio_initial_value = 1000,
        initial_position = 0,
        max_episode_duration = 128,
        max_position_duration = 128,
        verbose = 1,
        render_mode= "logs",
        random_start = False
    )

# Train Env에서 metric을 추가해줍니다.
test_env.unwrapped.add_metric('Sharpe ratio', lambda history : round(np.sum(np.sum(history['portfolio_return', -1])/np.std(history['portfolio_return', -1])),4))
test_env.unwrapped.add_metric('reward', lambda history : round(history['reward', -1], 4))
test_env.unwrapped.add_metric('Episode Lenght', lambda history : len(history['position']) )

print("=========================test=========================")
done, truncated = False, False
observation, info = test_env.reset()

model = PPO.load("ppo_chart_2")

pnl_list = []
for i in range(len(test_df)-test_env.unwrapped.windows-2):
    action, _states = model.predict(observation)
    observation, reward, done, truncated, info = test_env.step(action)
    if done or truncated:
        start_idx = info['idx']
        pnl_list.append(info['portfolio_valuation'])
        test_env.reset(start_idx=start_idx+1)
        test_env.unwrapped.save_for_render(log_name="ppo_test")
    
print(sum(pnl_list))

# Save for render

    
renderer = Renderer(render_logs_dir="render_logs")
renderer.run()