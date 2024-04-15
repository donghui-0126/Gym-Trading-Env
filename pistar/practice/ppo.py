# https://github.com/seungeunrho/minimalRL
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import os


df = pd.read_csv(r"C:\Users\user\Documents\GitHub\Gym-Trading-Env\pistar\data\SS00001.csv", parse_dates=["date"], index_col= "date")
df.dropna(inplace=True)

train_df = df.iloc[:-800].copy()
test_df = df.iloc[-800:].copy()

train_df['feature_close'] = (train_df['close'] - train_df['close'].min()) / (train_df['close'].max() - train_df['close'].min())
train_df["feature_volume"] = train_df['volume'].copy() / 100000
train_df["feature_RSI"] = train_df["RSI"].copy() / 100
train_df["feature_MACD"] = train_df["MACD"].copy()
train_df["feature_CCI"] = train_df["CCI"].copy() / 100
train_df["feature_ADX"] = train_df["ADX"].copy() 
train_df['feature_pct_change'] = train_df['close'].pct_change() * 10


def reward_function_when_execute(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", 0])

def sharpe_tanh_reward_function_when_execute(history):
    sharpe_ratio = np.sum(history['portfolio_return', -1])/np.std(history['portfolio_return', -1])
    reward = np.tanh(sharpe_ratio)
    return reward

def dynamic_feature_last_position_taken(history):
    return history['position', -1] / 10

def dynamic_feature_asset(history):
    return history['asset', -1] / 1000

def dynamic_feature_fiat(history):
    return history['fiat', -1] / 10

def dynamic_feature_step(history):
    return history['step', -1] / 128

# Train Env 구성
train_env = gym.make(
        "TradingEnv",
        name= "stock",
        df = train_df,
        positions = [-5,-3,-1,0,1,3,5],
        dynamic_feature_functions = [dynamic_feature_last_position_taken,  
                                    dynamic_feature_asset, 
                                    dynamic_feature_fiat,
                                    dynamic_feature_step],
        reward_function_when_execute = sharpe_tanh_reward_function_when_execute,
        windows = 30,
        trading_fees = 0.04/100,
        portfolio_initial_value = 1000,
        initial_position = 'random',
        max_episode_duration = 15,
        max_position_duration = 15,
        verbose = 1,
        render_mode= "logs",
        random_start = True
    )

# 모델을 불러옵니다.
class PPO(nn.Module):
    def __init__(self, 
                 action_num, 
                 input_dim,
                 learning_rate = 0.0001, 
                 gamma = 0.98,
                 lmbda = 0.95,
                 eps_clip = 0.05,
                 K_epoch= 5):
        super(PPO, self).__init__()
        self.action_num = action_num
        self.input_dim = input_dim
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.data = []
        
        self.fc1   = nn.Linear(self.input_dim+2,512)
        self.fc2 =  nn.Linear(512,256)
        self.fc_pi = nn.Linear(256,self.action_num)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, temperature, softmax_dim=-1):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logit = self.fc_pi(x)
        logit /= temperature
        prob = F.softmax(logit, dim=softmax_dim)
        return prob.view(x.size(0), -1)  # (2, 7)의 형태로 변환
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [],[],[],[],[],[]
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a, append_element = torch.tensor(np.array(s_lst), dtype=torch.float), \
                                            torch.tensor(np.array(a_lst), dtype=torch.int64), \
                                            torch.tensor(np.array(r_lst), dtype=torch.int64), \
                                            torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
                                            torch.tensor(np.array(done_lst), dtype=torch.float), \
                                            torch.tensor(np.array(prob_a_lst))
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self, temperature):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        step = a.shape[0]
        
        s = s.reshape(step, -1)
        s_prime = s_prime.reshape(step, -1)
        
        for i in range(self.K_epoch):
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(np.array(advantage_lst), dtype=torch.float)

            pi = self.pi(s, temperature=temperature, softmax_dim=-1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            policy_loss = -torch.min(surr1, surr2)
            value_loss =  F.smooth_l1_loss(self.v(s) , td_target.detach())
            loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def main(windows=10, feature_num=9, model_directory="model"):
    positions = [-5,-3,-1,0,1,3,5]
    action_num = len(positions)
    model = PPO(action_num=action_num, input_dim=windows*feature_num, K_epoch=5)
    
    env = gym.make(
        "TradingEnv",
        name= "stock",
        df = train_df,
        positions = positions,
        dynamic_feature_functions = [dynamic_feature_last_position_taken,  
                                    dynamic_feature_step],

        reward_function_when_execute = reward_function_when_execute,
        windows = windows,
        trading_fees = 0.04/100,
        portfolio_initial_value = 1000,
        initial_position = 0,
        max_episode_duration = 14,
        max_position_duration = 14,
        verbose = 1,
        render_mode= "logs",
        random_start = True)
    
    start_temperature = 10
    end_temperature = 1
    half_episodes = 5000  # 전체 에피소드의 절반
    for n_epi in range(20000):
        print(f'==================={n_epi}===================')
        s, _ = env.reset()
        mask = np.isnan(s)
        s[mask] = 0
    
        done = False
        
        if n_epi <= half_episodes:
            ratio = (n_epi - half_episodes) / half_episodes
            temperature = start_temperature - ratio * (start_temperature - end_temperature)
        else:
            temperature = end_temperature
        
        while not done:
            for t in range(15):
                
                s = s.reshape(1, -1)
                
                prob = model.pi(x = torch.from_numpy(s).float(), 
                                temperature=temperature,
                                softmax_dim=-1)
                
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)
                
                mask = np.isnan(s_prime)
                s_prime[mask] = 0
                
                s_prime = s_prime.reshape(1, -1)
                 
                model.put_data((s, a, r/100.0, s_prime, prob[:, a].item(), done))
                                
                if done or truncated:
                    break
                
            print("train")
            model.train_net(temperature)
            print("train finish")
            
    env.close()

if __name__ == '__main__':
    main()