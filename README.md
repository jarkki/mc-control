**mc-control** is a C++ library for solving stochastic dynamic optimization problems with *Monte Carlo optimal control*. It solves continuous state & continuous action problems by discretizing the continuous variables.

# Introduction
The library implements the two on-policy algorithms (exploring starts, sigma-soft policy) described in the 5th chapter of 

>Sutton, Richard S., and Andrew G. Barto. [*Reinforcement learning: An introduction*](http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html). MIT press, 1998.

While *approximate dynamic programming* methods like *fitted value iteration* can be the logical choice for continuous state & continuous action problems, they can be unstable and hard to implement due to the several layers of approximations. The Monte Carlo control algorithms can be very useful for checking the results obtained from dynamic programming methods. Depending on the nature of the problem, the Monte Carlo methods can be a better fit compared to other reinforcement learning methods like *Q-learning* and *fitted Q-iteration*. MC methods are also less prone to violations of the Markov property and do not need a model for the dynamics, only simulations or samples from interacting with the system.

The optimization problem considered is the stochastic dynamic optimization problem of maximizing the expected discounted rewards over either a finite or an infinite time horizon. The finite horizon problem is 

![](figures/eq_no_01.png?raw=true), 

where ![](figures/eq_no_02.png?raw=true) is a policy function, ![](figures/eq_no_03.png?raw=true) is a discount factor, ![](figures/eq_no_04.png?raw=true) is a reward function, ![](figures/eq_no_05.png?raw=true) is a stochastic state variable, ![](figures/eq_no_06.png?raw=true) is an action taken by the agent when at state ![](figures/eq_no_07.png?raw=true), following the policy ![](figures/eq_no_08.png?raw=true). ![](figures/eq_no_09.png?raw=true) denotes the time period. The state variable ![](figures/eq_no_10.png?raw=true) is assumed to be Markovian, which is why problems of this type are often called *Markov Decision Processes*.

## Curse of dimensionality

# Installation
## Dependencies
This library depends on two other libraries:

* Armadillo (for matrices, vectors and random number generation)
* Boost     (for boost::irange range-based iterator)

If the compiler cannot find either of the libraries, modify the [makefile](Makefile), which has variables for custom header and library search paths for these libraries (boost is header only).

## Compilation
`mc-control` is header-only library and uses some c++11 features. Just run `make` in the root directory to compile the example consumption model. Edit the [makefile](Makefile) if armadillo or boost is not found.


# Example
A classic example for a stochastic dynamic optimization problem in economics is the neoclassical consumption model where an agent splits her income into consumption and savings and seeks the savings policy that maximizes her expected discounted utility from consumption over an infinite time horizon:

![](figures/eq_no_11.png?raw=true)

![](figures/eq_no_12.png?raw=true)

s.t.

![](figures/eq_no_13.png?raw=true), (feasibility constraint),

The transition function for income is

![](figures/eq_no_14.png?raw=true)

with the action ![](figures/eq_no_15.png?raw=true) representing the amount to save, given the income.

Popular choice for the shock is log-normal distribution ![](figures/eq_no_16.png?raw=true) For utility function, ![](figures/eq_no_17.png?raw=true).


## Transforming the consumption model into reinforcement learning domain
The Monte Carlo optimal control belongs to the group of reinforcement learning algorithms. In the reinforcement learning domain the properties of interest are the 

1. State space
2. Actions
3. Transition function
4. Reward function
5. Simulating an episode from the model

The consumption model has a single continuous state variable: income ![](figures/eq_no_18.png?raw=true), now denoted as ![](figures/eq_no_19.png?raw=true)

The continuous action variable is the amount to save, given the income: ![](figures/eq_no_20.png?raw=true)

The output from the transition function is the next state ![](figures/eq_no_21.png?raw=true) that follows after first being in a state ![](figures/eq_no_22.png?raw=true) and then taking action:  ![](figures/eq_no_23.png?raw=true)

Reward function is the utility function ![](figures/eq_no_24.png?raw=true), or ![](figures/eq_no_25.png?raw=true)

# Algorithmic details
## Discretize the state space
With Monte Carlo control, the continuous stochastic state variables have to be discretized. `mc-control` uses the following method:

1. Draw samples from the continuous distribution of state variable
2. Divide the state space into bins and create discrete density function
3. Create inverse cumulative distribution function form the discrete density function
4. Use [inverse transform method](https://en.wikipedia.org/wiki/Inverse_transform_sampling) to sample from the resulting  discrete distribution

## Discretize the action space
The action space is discretized to a finite number of actions.

## Monte Carlo control with exploring starts
The algorithm, reproduced from Sutton & Barto (1998) chapter 5. 

![](figures/MC-ES.png?raw=true)

For infinite horizon problems, this algorithm reduces to randomly sampling the state-action space.

## Monte Carlo control with a soft policy
