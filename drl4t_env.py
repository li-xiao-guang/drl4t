import numpy as np
import gym
from gym import spaces
import random
import enum

class Actions(enum.Enum):
    Hold = 0
    Buy = 1
    Sell = 2
    
class DRL4TEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 data, 
                 indicator_columns=['SMARatio10', 'SMARatio20', 'MACD', 'BBP', 'CMF'],
                 price_column='Close',
                 sample_days=30, 
                 starting_balance=100000, 
                 commission_rate=0.001, 
                 random_on_reset=True):
        super(DRL4TEnv, self).__init__()

        self.indicator_columns = indicator_columns
        self.price_column = price_column
        self.sample_days = sample_days
        self.starting_balance = starting_balance
        self.commission_rate = commission_rate
        self.random_on_reset = random_on_reset

        self.data = dict(filter(lambda item: len(item[1]) > sample_days, data.items()))
        
        self.cur_episode = self.next_episode() if random_on_reset else 0
        self.cur_step = self.sample_days

        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, 
                                                high=np.inf, 
                                                shape=(len(self.cur_indicators.columns) * self.sample_days + 3,), 
                                                dtype=np.float16)
        self.reward_range = (-np.inf, np.inf)

        self.cash = self.starting_balance
        self.shares = 0
        
    @property
    def total_episodes(self):    
        return len(self.data)
    
    @property
    def cur_symbol(self):
        return list(self.data.keys())[self.cur_episode]

    @property
    def cur_data(self):
        return self.data[self.cur_symbol]

    @property
    def cur_indicators(self):
        return self.cur_data[self.indicator_columns]

    @property
    def total_steps(self):    
        return len(self.cur_data)
    
    @property
    def cur_close_price(self):
        return self.cur_data[self.price_column][self.cur_step]
    
    @property
    def cur_balance(self):
        return self.cash + (self.shares * self.cur_close_price)
    
    def next_episode(self):
        if self.random_on_reset:
            return random.randrange(0, self.total_episodes)
        else:
            return (self.cur_episode + 1) % self.total_episodes
                
    def next_observation(self, action):
        observation = []
        for i in range(self.sample_days, 0, -1):
            observation = np.append(observation, self.cur_indicators.values[self.cur_step - i + 1])
        return np.append(observation, [self.cash, self.shares * self.cur_close_price, action])
    
    def take_action(self, action):
        if action == Actions.Buy.value:
            if (self.shares == 0):
                price = self.cur_close_price * (1 + self.commission_rate)
                self.shares = int(self.cash / price / 100) * 100
                self.cash -= self.shares * price
        elif action == Actions.Sell.value:
            if (self.shares > 0):
                price = self.cur_close_price * (1 - self.commission_rate)
                self.cash += self.shares * price
                self.shares = 0
        
    def reset(self):
        self.cash = self.starting_balance
        self.shares = 0
        return self.next_observation(Actions.Hold.value)
    
    def step(self, action):
        balance = self.cur_balance

        self.cur_step += 1
        if self.cur_step == self.total_steps:
            self.cur_episode = self.next_episode()
            self.cur_step = self.sample_days

        self.take_action(action)

        obs = self.next_observation(action)
        reward = self.cur_balance - balance
        done = self.cur_step == self.total_steps - 1
        info = { 'Date'  : self.cur_data.index[self.cur_step].strftime('%Y-%m-%d'),
                 'Reward' : round(reward, 2),
                 'Symbol' : self.cur_symbol,
                 'Action' : Actions(action).name,
                 'Shares' : self.shares, 
                 'Close'  : round(self.cur_close_price, 2),
                 'Cash'   : round(self.cash, 2), 
                 'Total'  : round(self.cur_balance, 2) }
        
        if done:
            self.reset()

        return obs, reward, done, info