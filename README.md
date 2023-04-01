# Deep Reinforcement Learning for Trading (drl4t)

This repository includes all the scripts and sample data used in my blog series, Creating a Deep Learning Trading Model Prototype Step by Step.

## Table of Blog Contents
* [Acquiring data, the first step towards using machine learning for stock trading (drl4t-01)](https://lixiaoguang.medium.com/acquiring-data-the-first-step-towards-using-machine-learning-for-stock-trading-ml4t-001-afcafb338ad5?source=friends_link&sk=0c0d25b8ea3328a943a33c27c9e303d1)
* [Technical indicators, tools for predicting trading market trends (drl4t-02)](https://lixiaoguang.medium.com/technical-indicators-tools-for-predicting-trading-market-trends-ml4t-002-7117226e4ade?source=friends_link&sk=f9f3f040e373fa42e49db94f8f96157f)
* [Data preparation, where to get a list of all NYSE and NASDAQ stocks (drl4t-03)](https://lixiaoguang.medium.com/where-to-get-a-list-of-all-nyse-and-nasdaq-stocks-ml4t-003-31198c40405e?source=friends_link&sk=1518f02ccf5043c69ec01273152dd43b)
* [Create custom OpenAI Gym environment for Deep Reinforcement Learning (drl4t-04)](https://lixiaoguang.medium.com/create-custom-openai-gym-environment-for-deep-reinforcement-learning-drl-af2b2e3c830d?source=friends_link&sk=0cece979f82c6ed0eb263ffd36bb8d12)
* [Training DQN models for trading using PyTorch and Stable-Baselines3 (drl4t-05)](https://lixiaoguang.medium.com/training-dqn-models-for-trading-using-pytorch-and-stable-baselines3-ml4t-005-2c256373db7b?source=friends_link&sk=e2e8e9e9f5cc45f3f4ca86e79af15baf)
* [Flask makes it so easy to share your predictions (drl4t-06)](https://lixiaoguang.medium.com/flask-makes-it-so-easy-to-share-your-predictions-drl4t-006-ce9b32285413?source=friends_link&sk=6ef3a09bf22764e6e388627ba068ae7c)

## Libraries used
* [yfinance](https://github.com/ranaroussi/yfinance)
* [OpenAI Gym](https://github.com/openai/gym)
* [PyTorch](https://github.com/pytorch/pytorch)
* [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
* [Flask](https://github.com/pallets/flask)

## Websites used
* [NASDAQ](https://www.nasdaq.com/market-activity/stocks/screener)

## Sample Data
* nyse.csv: a list of 23 large companies whose stocks are traded on NYSE

## Example Model
* examples/nyse_dqn_model.pt: a trading model trained over 100,000 time steps 
