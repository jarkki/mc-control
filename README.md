**mc-control** is a C++ library for solving stochastic dynamic optimization problems with *Monte Carlo optimal control*. It solves continuous state & continuous action problems by discretizing the continuous variables.

Example discretized probability distribution for optimal savings problem (one state variable):

![Discretized probability distribution](figures/discrete_density.png)

Example value function contours with optimal policy for optimal savings problem:

![Optimal policy for optimal consumption problem](figures/optimal_policy.png)

# Introduction
The library implements the two on-policy algorithms (exploring starts, epsilon-soft policy) described in the 5th chapter of 

>Sutton, Richard S., and Andrew G. Barto. [*Reinforcement learning: An introduction*](http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html). MIT press, 1998.

While *approximate dynamic programming* methods like *fitted value iteration* can be the logical choice for continuous state & continuous action problems, they can be unstable and hard to implement due to the several layers of approximations. The Monte Carlo control algorithms can be very useful for checking the results obtained from dynamic programming methods. Depending on the nature of the problem, the Monte Carlo methods can be a better fit compared to other reinforcement learning methods like *Q-learning* and *fitted Q-iteration*. MC methods are also less prone to violations of the Markov property and do not need a model for the dynamics, only simulations or samples from interacting with the system.

The optimization problem considered is the stochastic dynamic optimization problem of finding a policy that maximizes the expected discounted rewards over either a finite or an infinite time horizon. The finite horizon problem is 

![](figures/eq_no_01.png?raw=true), 

where ![](figures/eq_no_02.png?raw=true) is a policy function, ![](figures/eq_no_03.png?raw=true) is a discount factor, ![](figures/eq_no_04.png?raw=true) is a reward function, ![](figures/eq_no_05.png?raw=true) is a stochastic state variable, ![](figures/eq_no_06.png?raw=true) is an action taken by the agent when at state ![](figures/eq_no_07.png?raw=true), following the policy ![](figures/eq_no_08.png?raw=true). ![](figures/eq_no_09.png?raw=true) denotes the time period. The state variable ![](figures/eq_no_10.png?raw=true) is assumed to be Markovian, which is why problems of this type are often called *Markov Decision Processes*.

# Installation
## Dependencies
This library depends on three other libraries:

* [Armadillo](http://arma.sourceforge.net) for matrices, vectors and random number generation
* [Boost](http://www.boost.org/) for boost::irange range-based iterator

For plotting you also need

* Python + numpy + [matplotlib](http://matplotlib.org/)

If the compiler cannot find Armadillo or Boost, modify the [makefile](Makefile), which has variables for custom header and library search paths for these libraries (boost is header only).

## Compilation
`mc-control` is a header-only library and uses some c++11 features. Just run `make` in the root directory to compile the example optimal savings model. Edit the [makefile](Makefile) if armadillo or boost is not found.


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

More details:
- Stachurski, John. *Economic dynamics: theory and computation*. MIT Press, (2009).
- Stokey, Nancy, and R. Lucas. *Recursive Methods in Economic Dynamics* Harvard University Press (1989).

Implementation can be found here: [examples/optgrowth.cpp](examples/optgrowth.cpp).

## Solving the dynamic problem
Dynamic optimization problem such as the optimal consumption/savings can be solved with the help of the recursive [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation):

![](figures/eq_no_18.png?raw=true)

The Bellman equation represents the value ![](figures/eq_no_19.png?raw=true) of being in a state ![](figures/eq_no_20.png?raw=true) and following policy ![](figures/eq_no_21.png?raw=true).

For the optimal savings problem the Bellman equation represents the rewards/returns as

![](figures/eq_no_22.png?raw=true).

## Discretizing the state and action variables
To discretize the state variable ![](figures/eq_no_23.png?raw=true), we go through these steps:

1. Draw samples from the continuous distribution of state variable
2. Divide the state space into bins and create discrete density(mass) function
3. Create inverse cumulative distribution function form the discrete density function
4. Use [inverse transform method](https://en.wikipedia.org/wiki/Inverse_transform_sampling) to sample from the resulting  discrete distribution

(Note that since we have the density function available for the state variable, instead of sampling we could use the density function directly to discretize the space.)

# Implementation details
Any model has to be derived from the base model struct:

```c++
/*! Abstract base struct for the models
*
*/
struct Model{

    /*! next_state = f(state, action)*/
    virtual vec transition(const vec & state, const double & action) const = 0;

    /*! Samples the transition funciton n times*/
    virtual mat sample_transitions(const double & action, size_t n) const = 0;

    /*! Reward from being in a state, taking action and ending in next_state */
    virtual double reward (const vec & state_value, const double & action_value, const vec & next_state_value) const = 0;

    /*! Returns true if it is possible to take the action from this state */
    virtual bool constraint(const double & action, const vec & state) const{
    return true;
    };
};
```

Then one of the two episode generating functions has to be implemented:
```c++
// For soft policies
tuple<uvec,uvec,vec> episode_soft_pol(const DiscretizedOptimalGrowthModel & discrete_model,  const uvec & pol);

// For exploring starts
tuple<uvec,uvec,vec> episode_es(const DiscretizedOptimalGrowthModel & discrete_model,  const size_t & state,  const size_t & action, const  uvec & pol);
```
The episode generating functions returns a three-tuple of all actions, states and returns occurring during the episode.

For a full example implementing the optimal savings model, see [examples/optgrowth.cpp](examples/optgrowth.cpp).


<!-- Let's discretize the state ![](figures/eq_no_24.png?raw=true) into 30 bins in the interval ![](figures/eq_no_25.png?raw=true) and action variable ![](figures/eq_no_26.png?raw=true) into 10 values. With 100k samples from ![](figures/eq_no_27.png?raw=true) the discrete approximation to the state-action density looks like this: -->

<!-- ![Discretized probability distribution](figures/discrete_density.png) -->

# The two implemented algorithms
See chapter 5. in [*Reinforcement learning: An introduction*](http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html) for details.

## Monte Carlo control with exploring starts
The algorithm, reproduced from Sutton & Barto (1998) chapter 5. 

![](figures/mc-es.png?raw=true)

For infinite horizon problems, this algorithm reduces to randomly sampling the state-action space.

## Monte Carlo control with an soft policy (epsilon greedy)

![](figures/mc-eps-greedy.png?raw=true)




