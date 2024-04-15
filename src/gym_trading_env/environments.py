import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import datetime
import glob
from pathlib import Path    
import random

from collections import Counter
from .utils.history import History
from .utils.portfolio import Portfolio, TargetPortfolio

import tempfile, os
import warnings
warnings.filterwarnings("error")

def basic_reward_function(history : History):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

def basic_reward_function_when_execute(history : History):
    position_duration = history["step", -1]
    return np.log(history["portfolio_valuation", -2] / history["portfolio_valuation", -(position_duration+1)])

def sharpe_tanh_reward_function_when_execute(history):
    sharpe_ratio = np.sum(history['portfolio_return', -1])/np.std(history['portfolio_return', -1])
    reward = np.tanh(sharpe_ratio)
    return reward

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


class TradingEnv(gym.Env):
    metadata = {'render_modes': ['logs']}
    def __init__(self,
                df : pd.DataFrame,
                positions : list = [0, 1],
                dynamic_feature_functions = [dynamic_feature_last_position_taken, 
                                             dynamic_feature_real_position, 
                                             dynamic_feature_asset, 
                                             dynamic_feature_fiat],
                
                reward_function_when_execute = sharpe_tanh_reward_function_when_execute,
                windows = None,
                trading_fees = 0,
                portfolio_initial_value = 1000,
                initial_position ='random',
                max_episode_duration = 'max',
                max_position_duration = 'max',
                verbose = 1,
                name = "Stock",
                render_mode= "logs",
                random_start = True
                ):
        
        self.max_episode_duration = max_episode_duration
        self.name = name
        self.verbose = verbose

        self.positions = positions
        self.dynamic_feature_functions = dynamic_feature_functions
        self.reward_function_when_execute = reward_function_when_execute
        self.windows = windows
        self.trading_fees = trading_fees
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.initial_position = initial_position
        self.initial_position in self.positions or self.initial_position == 'random'
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.max_episode_duration = max_episode_duration
        self.max_position_duration = max_position_duration
        self.render_mode = render_mode
        self.random_start = random_start
        
        self.max_position = max(self.positions)
        self.min_position = min(self.positions)
        
        self._set_df(df)   
        
        self.action_space = spaces.Discrete(len(positions))
        
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape = [self._nb_features]
        )
            
        if self.windows is not None:
            self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape = [self.windows, self._nb_features]
            )
        
        self.log_metrics = []


    def _set_df(self, df):
        df = df.copy()
        self._features_columns = [col for col in df.columns if "feature" in col]
        self._info_columns = list(set(list(df.columns) + ["close"]) - set(self._features_columns))
        self._nb_features = len(self._features_columns)
        self._nb_static_features = self._nb_features

        for i  in range(len(self.dynamic_feature_functions)):
            df[f"dynamic_feature__{i}"] = 0
            self._features_columns.append(f"dynamic_feature__{i}")
            self._nb_features += 1

        self.df = df
        self._obs_array = np.array(self.df[self._features_columns], dtype= np.float32)
        self._info_array = np.array(self.df[self._info_columns])
        self._price_array = np.array(self.df["close"])

    
    def _get_price(self, delta = 0):
        return self._price_array[self._idx + delta]
    
    def _get_obs(self):
        for i, dynamic_feature_function in enumerate(self.dynamic_feature_functions):
            self._obs_array[self._idx, self._nb_static_features + i] = dynamic_feature_function(self.historical_info)

        if self.windows is None:
            _step_index = self._idx
        else: 
            _step_index = np.arange(self._idx + 1 - self.windows , self._idx + 1)
            
        return self._obs_array[_step_index]

    
    def reset(self, seed = None, options=None, start_idx=None):
        super().reset(seed = seed)
        self._step = 0
        self._position = np.random.choice(self.positions) if self.initial_position == 'random' else self.initial_position
            
        self._idx = 0
        
        if start_idx != None:
            self._idx = start_idx 
        
        if self.random_start == True:
            self._idx = np.random.randint(low = self._idx, 
                                          high = len(self.df)-self.max_episode_duration-self._idx)
            
        self._portfolio  = TargetPortfolio(
            position = self._position,
            value = self.portfolio_initial_value,
            price = self._get_price(),
            init_value=self.portfolio_initial_value
        )
        
        
        self._take_action(self._position, is_execute=False, previous_position=0)
        price = self._get_price(delta=1) 
        
        portfolio_value = self._portfolio.valorisation(price)
        portfolio_distribution = self._portfolio.get_portfolio_distribution()
        
        portfolio_return_1_step =  np.log(abs(portfolio_value) / self.portfolio_initial_value)
        portfolio_return = np.array([portfolio_return_1_step])
        
        
        self.historical_info = History(max_size= len(self.df))
        
        self.historical_info.set(
            idx = self._idx,
            step = self._step,
            date = self.df.index.values[self._idx],
            position = self._position,
            data =  dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation = self.portfolio_initial_value,
            portfolio_distribution = self._portfolio.get_portfolio_distribution(),
            reward = 0,
            asset = self._portfolio.get_portfolio_distribution()["asset"],  
            fiat = self._portfolio.get_portfolio_distribution()["fiat"],
            execute = False,
            portfolio_return = portfolio_return,
        )
        
        
        return self._get_obs(), self.historical_info[0]


    def _trade(self, position, previous_position, price = None):
        self._portfolio.trade_to_position(
            position, 
            price = self._get_price() if price is None else price, 
            trading_fees = self.trading_fees,
            precious_position=previous_position,
            init_value=self.portfolio_initial_value)
        
        self._position = position
        return

    def _take_action(self, position, is_execute, previous_position):
        if position != self._position:
            if is_execute == True:
                self._trade(position=0, previous_position=previous_position)
            elif is_execute == False:
                self._trade(position, previous_position=previous_position)
    
    def step(self, position_index = None):   
        previous_position = self.historical_info["position",-1]
        current_position = self.positions[position_index]
        
        is_execute = (previous_position * current_position < 0)
        
        if previous_position !=0 and current_position == 0: 
            is_execute=True

        if isinstance(self.max_position_duration,int) and self._step >= self.max_position_duration - 1:
            is_execute = True

        self._idx += 1
        self._step += 1

        # 다음 step의 가격으로 계산
        self._take_action(self.positions[position_index], is_execute, previous_position)
        
        price = self._get_price() 
        portfolio_value = self._portfolio.valorisation(price)
        portfolio_distribution = self._portfolio.get_portfolio_distribution()
        
        portfolio_return_1_step =  np.log(abs(portfolio_value) / self.historical_info["portfolio_valuation", -1])
        portfolio_return = np.append(self.historical_info["portfolio_return", -1], portfolio_return_1_step)
        

        self.historical_info.add(
            idx = self._idx,
            step = self._step,
            date = self.df.index.values[self._idx],
            position = self._position,
            data =  dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation = portfolio_value,
            portfolio_distribution = portfolio_distribution, 
            reward = 0,
            asset = portfolio_distribution["asset"],
            fiat = portfolio_distribution["fiat"],
            execute = is_execute,
            portfolio_return = portfolio_return,
            )
        
        done, truncated = False, False
            
        if is_execute:
            reward = self.reward_function_when_execute(self.historical_info)
            self.historical_info["reward", -1] = reward
            done = True

            
        if done or truncated:
            self.calculate_metrics()
            self.log()
                            
        return self._get_obs(),  self.historical_info["reward", -1], done, truncated, self.historical_info[-1]

    def add_metric(self, name, function):
        self.log_metrics.append({
            'name': name,
            'function': function
        })
        
    def calculate_metrics(self):
        self.results_metrics = {
            "Market Return" : f"{100*(self.historical_info['data_close', -1] / self.historical_info['data_close', 0] -1):5.2f}%",
            "Portfolio Return" : f"{100*(self.historical_info['portfolio_valuation', -1] / self.historical_info['portfolio_valuation', 0] -1):5.2f}%",
        }
        for metric in self.log_metrics:
            self.results_metrics[metric['name']] = metric['function'](self.historical_info)
    
    def get_metrics(self):
        return self.results_metrics
    
    def log(self):
        if self.verbose > 0:
            text = ""
            for key, value in self.results_metrics.items():
                text += f"{key} : {value}   |   "
            print(text)