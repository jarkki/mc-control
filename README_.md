# mc-control
A C++ library for solving continuous state, continuous action stochastic dynamic optimization problems with **Monte Carlo optimal control**.

The library implements the two on-policy algorithms described in the 5th chapter of 

>Sutton, Richard S., and Andrew G. Barto. *Reinforcement learning: An introduction*. MIT press, 1998.

found at [here](http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html).

# Introduction
The optimization problem considered here is the stochastic dynamic optimization problem of maximizing the expected discounted rewards over either a finite or an infinite time horizon. The finite horizon problem is 

$\underset{\pi}{\text{max}} \ E \left[\sum_{t=0}^{T}\gamma^tR_{t}^{\pi}(S_t, A_t^{\pi}(S_t))\right]$, 

where $\pi$ is a policy function, $\gamma$ is a discount factor, $R$ is a reward function, $S$ is a stochastic state variable, $A^{\pi}(S)$ is an action taken by the agent when at state $S$, following the policy $\pi$. $t \in \{1,\ldots,T\}$ denotes the time period. The state variable $S_t$ is assumed to be Markovian, which is why problems of this type are often called *Markov Decision Processes*.

# Installation

# Example
A classic example in economics is the neoclassical consumption model where an agent splits her income into consumption and savings and seeks the savings policy that maximizes her expected discounted utility from consumption over time.

Income $y_t$ is split into savings $k_t$ and consumption $c_t$. Income process $y=f(k,W)$ can be for example

$y = k^{\alpha}W,$

where $W$ is an iid. random shock.
	
Agent wants to maximize utility from consumption. Agent searches for optimal saving function $a(y_t)$ given the income, that maximizes the infinite horizon expected utility from consumption:

$\underset{k_t}{\text{max}} \ E \left[\sum_{t=0}^{\infty}\gamma^tU(c_t)\right]$

$\Leftrightarrow \underset{k_t}{\text{max}} \ E \left[\sum_{t=0}^{\infty}\gamma^tU(y_t - a(y_t))\right],$

s.t.

$0 \leq a(y) \leq y$, (feasibility constraint),

where $c_t = y_t - k_t = y_t - a(y_t)$ and $\gamma^t$ is the discount factor.

The transition function for income is

$y_{t+1} = a(y_t)^{\alpha}W_{t+1},$

with $k_t = a(y_t) = \theta y_t$, where $\theta$ is the savings rate. 

Popular choice for the shock is log-normal distribution $W \sim e^{N(0,1)}$


## Transforming the consumption model into reinforcement learning domain
The Monte Carlo optimal control belongs to the group of reinforcement learning algorithms. In the reinforcement learning domain the properties of interest are the 

1. State space
2. Actions
3. Transition function
4. Reward function
5. Simulating an episode from the model

The consumption model has a single continuous state variable: income $y_t$, now denoted as $s_t.$

The continuous action variable is the amount to save, given the income: $k_t = a(y_t)$, now $a(s_t).$

The output from the transition function is the next state $s_{t+!}$ that follows after first being in a state $s_t$ and then taking action $a(s_t)$: $y_{t+1} = a(y_t)^{\alpha}W_{t+1}$, now $s_{t+1} = a(s_t)^{\alpha}W_{t+1}.$

Reward function is the utility function $U(c_t)$, now $U(s_t-a_t).$

# Algorithmic details
## Discretize the state space
With Monte Carlo control, the continuous stochastic state variables have to be discretized. `mc-control` uses the following method:

1. Draw samples from the continuous distribution of state variable
2. Divide the state space into bins and create discrete density function
3. Create inverse cumulative distribution function form the discrete density function
4. Use [inverse transform method](https://en.wikipedia.org/wiki/Inverse_transform_sampling) to sample from the resulting  discrete distribution

## Discretize the action space
The action space is discretized to a finite number of actions.

## The Monte Carlo control with exploring starts
The algorithm, reproduced from Sutton & Barto (1998) chapter 5. is 

![](figures/MC-ES.png?raw=true)

