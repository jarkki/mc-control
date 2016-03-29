# mc-control
A C++ library for solving continuous state, continuous action stochastic dynamic optimization problems with **Monte Carlo optimal control**.

# Introduction
The library implements the two on-policy algorithms described in the 5th chapter of 

>Sutton, Richard S., and Andrew G. Barto. *Reinforcement learning: An introduction*. MIT press, 1998.

found at [here](http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html).

The optimization problem considered is the finite period stochastic dynamic optimization problem (and its infinite horizon version)

![](figures/eq_no_01.png?raw=true), 

where ![](figures/eq_no_02.png?raw=true) is a policy function, ![](figures/eq_no_03.png?raw=true) is a discount factor, ![](figures/eq_no_04.png?raw=true) is a reward function, ![](figures/eq_no_05.png?raw=true) is a stochastic state variable, ![](figures/eq_no_06.png?raw=true) is an action taken by the agent when at state ![](figures/eq_no_07.png?raw=true), following the policy ![](figures/eq_no_08.png?raw=true). ![](figures/eq_no_09.png?raw=true) denotes the time period.

# Installation

# Example

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

