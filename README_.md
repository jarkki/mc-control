# mc-control
A C++ library for solving continuous state, continuous action stochastic dynamic optimization problems with **Monte Carlo optimal control**.

# Introduction
The library implements the two on-policy algorithms described in the 5th chapter of 

>Sutton, Richard S., and Andrew G. Barto. *Reinforcement learning: An introduction*. MIT press, 1998.

found at [here](http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html).

The optimization problem considered is the finite period stochastic dynamic optimization problem (and its infinite horizon version)

$\underset{\pi}{\text{max}} \ E \left[\sum_{t=0}^{T}\gamma^tR_{t}^{\pi}(S_t, A_t^{\pi}(S_t))\right]$, 

where $\pi$ is a policy function, $\gamma$ is a discount factor, $R$ is a reward function, $S$ is a stochastic state variable, $A^{\pi}(S)$ is an action taken by the agent when at state $S$, following the policy $\pi$. $t \in \{1,\ldots,T\}$ denotes the time period.

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

