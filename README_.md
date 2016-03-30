**mc-control** is a C++ library for solving stochastic dynamic optimization problems with *Monte Carlo optimal control*. It solves continuous state & continuous action problems by discretizing the continuous variables.

# Introduction
The library implements the two on-policy algorithms (exploring starts, sigma-soft policy) described in the 5th chapter of 

>Sutton, Richard S., and Andrew G. Barto. [*Reinforcement learning: An introduction*](http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html). MIT press, 1998.

While *approximate dynamic programming* methods like *fitted value iteration* can be the logical choice for continuous state & continuous action problems, they can be unstable and hard to implement due to the several layers of approximations. The Monte Carlo control algorithms can be very useful for checking the results obtained from dynamic programming methods. Depending on the nature of the problem, the Monte Carlo methods can be a better fit compared to other reinforcement learning methods like *Q-learning* and *fitted Q-iteration*. MC methods are also less prone to violations of the Markov property and do not need a model for the dynamics, only simulations or samples from interacting with the system.

The optimization problem considered is the stochastic dynamic optimization problem of maximizing the expected discounted rewards over either a finite or an infinite time horizon. The finite horizon problem is 

$\underset{\pi}{\text{max}} \ E \left[\sum_{t=0}^{T}\gamma^tR_{t}^{\pi}(S_t, A_t^{\pi}(S_t))\right]$, 

where $\pi$ is a policy function, $\gamma$ is a discount factor, $R$ is a reward function, $S$ is a stochastic state variable, $A^{\pi}(S)$ is an action taken by the agent when at state $S$, following the policy $\pi$. $t \in \{1,\ldots,T\}$ denotes the time period. The state variable $S_t$ is assumed to be Markovian, which is why problems of this type are often called *Markov Decision Processes*.

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

$\underset{k_t}{\text{max}} \ E \left[\sum_{t=0}^{\infty}\gamma^tU(c_t)\right]$

$\Leftrightarrow \underset{k_t}{\text{max}} \ E \left[\sum_{t=0}^{\infty}\gamma^tU(y_t - a(y_t))\right],$

s.t.

$0 \leq a(y) \leq y$, (feasibility constraint),

The transition function for income is

$y_{t+1} = a(y_t)^{\alpha}W_{t+1},$

with the action $k_t = a(y_t)$ representing the amount to save, given the income.

Popular choice for the shock is log-normal distribution $W \sim e^{N(0,1)}.$ For utility function, $U(c) = 1-e^{-\theta c}$.


## Transforming the consumption model into reinforcement learning domain
The Monte Carlo optimal control belongs to the group of reinforcement learning algorithms. In the reinforcement learning domain the properties of interest are the 

1. State space
2. Actions
3. Transition function
4. Reward function
5. Simulating an episode from the model

The consumption model has a single continuous state variable: income $y_t$, now denoted as $s_t.$

The continuous action variable is the amount to save, given the income: $k_t = a(s_t).$

The output from the transition function is the next state $s_{t+!}$ that follows after first being in a state $s_t$ and then taking action:  $s_{t+1} = a(s_t)^{\alpha}W_{t+1}.$

Reward function is the utility function $U(c_t)$, or $U(s_t-a_t).$

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
