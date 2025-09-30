## Stochastic Optimization and Control & Reinforcement Learning  

Stochastic control is a mathematical framework used to make optimal decisions in systems influenced by random noise.

### Objective
The goal is to implement, analyze, and compare various stochastic control techniques—from classical methods to modern machine learning approaches—with applications to real-world problems in portfolio optimization, risk management, option pricing, dynamic asset allocation, and algorithmic trading.

The project addresses both classical and modern techniques in stochastic control, including: 
•	• Convex optimization in stochastic environments 
•	• Discrete-time stochastic control 
•	• Continuous-time stochastic control 
•	• Reinforcement learning methods

**Quantitative Finance Applications:** 
Portfolio optimization, algorithmic trading strategies, asset allocation, and risk control.

We implement and analyze practical problems in quantitative finance, such as portfolio optimization. 
The main focus is on applying advanced topics in portfolio management, including;
- Reinforcement learning approaches 
- Mean-field theory


### Portfolio optimization techniques
- Mean-variance optimization
- Robust optimization

[Mean-Variance Optimization.ipynb](Mean-Variance_Optimization.py)


### Reinforcement Learning:
Training agents to learn optimal policies for trading, portfolio management, and risk mitigation through interaction with financial environments.

## Reinforcement Learning for Quantitative Trading

This project includes a reinforcement learning framework for developing and evaluating trading strategies in financial markets.

- 📄 [RL_Trading.py](https://github.com/mzallaghi4/Stochastic-Optimization-and-Control/blob/master/ReinforcementLearning/RL_Trading.py): Core implementation of the RL agent and trading environment.


## Reinforcement Learning for Portfolio Management

Portfolio management is the process of selecting, monitoring, and adjusting a collection of financial assets (stocks, bonds, ETFs, etc.) to meet specific investment goals, such as maximizing returns or minimizing risk over time.

Reinforcement Learning offers a dynamic, data-driven approach that enables an agent to learn optimal allocation strategies by interacting with the market environment.

#### RL Algorithms for Portfolio Management

**DDPG (Deep Deterministic Policy Gradient)**

DDPG is an off-policy, model-free, actor-critic algorithm designed for continuous action spaces.
- [DDPG Algorithm Implementation](https://github.com/mzallaghi4/Stochastic-Optimization-and-Control/blob/master/ReinforcementLearning/DDPG.py): Deep Deterministic Policy Gradient method for continuous action trading environments.
  
**SAC (Soft Actor-Critic)**

SAC is an off-policy, model-free actor-critic algorithm based on the maximum entropy RL framework.

- [SAC Algorithm Implementation](https://github.com/mzallaghi4/Stochastic-Optimization-and-Control/blob/master/ReinforcementLearning/SAC.py): Soft Actor-Critic algorithm offering stable and efficient learning for portfolio management.
