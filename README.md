# mc-control
A C++ library for solving continuous state, continuous action stochastic dynamic optimization problems with **Monte Carlo optimal control**.

The library implements the two on-policy algorithms described in the 5th chapter of 

>Sutton, Richard S., and Andrew G. Barto. *Reinforcement learning: An introduction*. MIT press, 1998.

found at [here](http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html).

# Introduction
The optimization problem considered here is the stochastic dynamic optimization problem of maximizing the expected discounted rewards over either a finite or an infinite time horizon. The finite horizon problem is 

![](figures/eq_no_01.png?raw=true), 

where ![](figures/eq_no_02.png?raw=true) is a policy function, ![](figures/eq_no_03.png?raw=true) is a discount factor, ![](figures/eq_no_04.png?raw=true) is a reward function, ![](figures/eq_no_05.png?raw=true) is a stochastic state variable, ![](figures/eq_no_06.png?raw=true) is an action taken by the agent when at state ![](figures/eq_no_07.png?raw=true), following the policy ![](figures/eq_no_08.png?raw=true). ![](figures/eq_no_09.png?raw=true) denotes the time period. The state variable ![](figures/eq_no_10.png?raw=true) is assumed to be Markovian, which is why problems of this type are often called *Markov Decision Processes*.

# Installation

# Example
A classic example in economics is the neoclassical consumption model where an agent splits her income into consumption and savings and seeks the savings policy that maximizes her expected discounted utility from consumption over time.

Income ![](figures/eq_no_11.png?raw=true) is split into savings ![](figures/eq_no_12.png?raw=true) and consumption ![](figures/eq_no_13.png?raw=true). Income process ![](figures/eq_no_14.png?raw=true) can be for example

![](figures/eq_no_15.png?raw=true)

where ![](figures/eq_no_16.png?raw=true) is an iid. random shock.
	
Agent wants to maximize utility from consumption. Agent searches for optimal saving function ![](figures/eq_no_17.png?raw=true) given the income, that maximizes the infinite horizon expected utility from consumption:

![](figures/eq_no_18.png?raw=true)

![](figures/eq_no_19.png?raw=true)

s.t.

![](figures/eq_no_20.png?raw=true), (feasibility constraint),

where ![](figures/eq_no_21.png?raw=true) and ![](figures/eq_no_22.png?raw=true) is the discount factor.

The transition function for income is

![](figures/eq_no_23.png?raw=true)

with ![](figures/eq_no_24.png?raw=true), where ![](figures/eq_no_25.png?raw=true) is the savings rate. 

Popular choice for the shock is log-normal distribution ![](figures/eq_no_26.png?raw=true)


## Transforming the consumption model into reinforcement learning domain
The Monte Carlo optimal control belongs to the group of reinforcement learning algorithms. In the reinforcement learning domain the properties of interest are the 

1. State space
2. Actions
3. Transition function
4. Reward function
5. Simulating an episode from the model

The consumption model has a single continuous state variable: income ![](figures/eq_no_27.png?raw=true), now denoted as ![](figures/eq_no_28.png?raw=true)

The continuous action variable is the amount to save, given the income: ![](figures/eq_no_29.png?raw=true), now ![](figures/eq_no_30.png?raw=true)

The output from the transition function is the next state ![](figures/eq_no_31.png?raw=true) that follows after first being in a state ![](figures/eq_no_32.png?raw=true) and then taking action ![](figures/eq_no_33.png?raw=true): ![](figures/eq_no_34.png?raw=true), now ![](figures/eq_no_35.png?raw=true)

Reward function is the utility function ![](figures/eq_no_36.png?raw=true), now ![](figures/eq_no_37.png?raw=true)

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

