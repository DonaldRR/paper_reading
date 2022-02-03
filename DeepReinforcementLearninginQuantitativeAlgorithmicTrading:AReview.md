# [Deep Reinforcement Learning in Quantitative Algorithmic Trading: A Review](https://arxiv.org/pdf/2106.00123.pdf)

## 1. Intro

## 2. Literature Review

### 2.1 Critic-only DRL
Cons: 
- DQN aims to solve a discrete action space problem
- Critic-only approaches are sensible to the reward signals from the environment

#### 2.1.1 [Application of Deep Reinforcement Learning on Automated Stock Trading](https://sci-hub.se/10.1109/icsess47205.2019.9040728)
Deep Recurrent Q-Net is proposed - A second target network can stablize the process. 

Cons:
1. Only one stock is used for benchmarking, the generability is loosely guaranteed.
2. The reward function is weak (Shape ratio could be used as a quick fix)
3. Constrained actions
4. Only raw data is used - technical analysis could be combined

#### 2.1.2 [Financial Trading as a Game: A Deep Reinforcement Learning Approach](https://arxiv.org/pdf/1807.02787.pdf)
Compared to the work `2.1.1`, it has improvements on:
1. Technical indicators are added
2. Memory replay
3. **Action augmentation**
4. Longer sequence as input

Cons:
1. Reward function still not address the risks
2. Limited action space

#### 2.1.3 [Adaptive Stock Trading Strategies with Deep Reinforcement Learning Methods](https://sci-hub.se/10.1016/j.ins.2020.05.066)
GRU is introduced for feature extraction, the whole work is denoted as GDQN. 

`The motivation for introducing this is the fact that stock market
movements cannot reveal the patterns or features behind the dynamic states of the market. In
order to achieve that, the authors present an architecture with an update gate and reset gate
function for the neural network.`

Pros:
1. A new reward function - Sortino ratio is used to better handle risk.
   
#### 2.1.4 [Deep Reinforcement Learning for Trading](https://arxiv.org/pdf/1911.10107.pdf)
The volatility of the market is incorporated in order to also consider the risk.

### 2.2 Actor-only DRL
A direct mapping from state to action is learnt, thus the action space is continuous, while it needs longer time to train.

#### 2.2.1 [Deep Direct Reinforcement Learning for Financial Signal Representation and Trading](http://cslt.riit.tsinghua.edu.cn/mediawiki/images/a/aa/07407387.pdf)

Propose a model that simultaneously senses the environment and makes recurrent decisions in an online working mode. 

Use neural network for feature extraction to reduce the uncertainty of the input data. 

A variant of Backpropagation through time(BTT) - task-aware BPTT handles the problem of vanishing gradients.

#### 2.2.2 [Quantitative Trading on Stock Market Based on Deep Reinforcement Learning](https://sci-hub.se/10.1109/ijcnn.2019.8851831)

Analyze the basic structure LSTM and fully-connected one as well as the effect of different compositions of technical indicators.

#### 2.2.3 [Enhancing Time Series Momentum Strategies Using Deep Neural Networks](https://sci-hub.se/10.2139/ssrn.3369195)

Focus more on the portfolio management.

### 2.3 Actor-critic DRL

The actor chooses a proper action based on the state while the critic measures how good the action is. 

The effectiveness of the actor-critic can be contributed to it's capability to deal with complex environments. 

While it suffers from unstability training - easy to crash to poor policy.

#### 2.3.1 [Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy](https://deliverypdf.ssrn.com/delivery.php?ID=991004073021021078082002067107082091096025095076029067119000122024069084005002006072121042119015047112061082126078112015075011119066064011050070079030103119026004099046054032120014085117093023092097029071119101082006071090085103000095028016093067098127&EXT=pdf&INDEX=TRUE)

Propose an ensemble system containing 3 models: PPO, A2C and DDPG. 

The ensemble can make the trading more robust and reliable for different situations and maximize return objective with risk constraints.

One more degree of freedom for such system is that features of whole bunch of stock is merge which enables the model to choose which stock to buy/sell. 

The intuition behind this work is, some models are sensitive to certain situations while others are not. A sole agent is chosen based on the validation performance over time.

#### 2.3.2 [Practical Deep Reinforcement Learning Approach for Stock Trading](https://arxiv.org/pdf/1811.07522.pdf)

Investigate the DDPG algorithm.

#### 2.3.3 [Stock Trading Bot Using Deep Reinforcement Learning](https://sci-hub.se/10.1007/978-981-10-8201-6_5)

Combine sentiment analysis and prove it learns some tricks. 

It tries to predict the sentiment of financial news with RCNN. 

3 variants of reward functions are tried:
1. difference between RL-agent asset value and stagnant asset value
2. difference between the cost at which the stocks are sold and the cost at which the stocks were bought
3. a binary reward representing if the action was profitable or not

Only the laer one work.

## 3. Conclusions

The environment is so complex that many aspects can not be modeled and then those algorithms would fail. While for trading patterns in a small timeframes, they can still be exploited.

