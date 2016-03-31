**mc-control** is a C++ library for solving stochastic dynamic optimization problems with *Monte Carlo optimal control*. It solves continuous state & continuous action problems by discretizing the continuous variables.

Example value function contours with optimal policy for optimal savings problem:

![Optimal policy for optimal consumption problem](figures/optimal_policy.png)

Example discretized probability distribution for optimal savings problem (one state variable):

![Discretized probability distribution](figures/discrete_density.png)

# Introduction
The library implements the two on-policy algorithms (exploring starts, epsilon-soft policy) described in the 5th chapter of 

>Sutton, Richard S., and Andrew G. Barto. [*Reinforcement learning: An introduction*](http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html). MIT press, 1998.

The optimization problem considered is the stochastic dynamic optimization problem of finding a policy that maximizes the expected discounted rewards over either a finite or an infinite time horizon. The finite horizon problem is 

$\underset{\pi}{\text{max}} \ \mathbb{E} \left[\sum\limits_{t=0}^{T}\gamma^tR_{t}^{\pi}(S_t, A_t^{\pi}(S_t))\right]$, 

where $\pi$ is a policy function, $\gamma$ is a discount factor, $R$ is a reward function, $S$ is a stochastic state variable, $A^{\pi}(S)$ is an action taken by the agent when at state $S$, following the policy $\pi$. $t \in \{1,\ldots,T\}$ denotes the time period. The state variable $S_t$ is assumed to be Markovian, which is why problems of this type are often called *Markov Decision Processes*.

While *approximate dynamic programming* methods like *fitted value iteration* can be the logical choice for continuous state & continuous action problems, they can be unstable and hard to implement due to the several layers of approximations. The Monte Carlo control algorithms can be very useful for checking the results obtained from dynamic programming methods. Depending on the nature of the problem, the Monte Carlo methods can be a better fit compared to other reinforcement learning methods like *Q-learning* and *fitted Q-iteration*. MC methods are also less prone to violations of the Markov property and do not need a model for the dynamics, only simulations or samples from interacting with the system.

# Installation
## Dependencies
This library depends on three other libraries:

* [Armadillo](http://arma.sourceforge.net) for matrices, vectors and random number generation
* [Boost](http://www.boost.org/) for boost::irange range-based iterator

For plotting you also need

* Python + numpy + [matplotlib](http://matplotlib.org/)

## Compilation
`mc-control` is a header-only library and uses some c++11 features. Just run `make` in the root directory to compile the example optimal savings model. 

If the compiler cannot find Armadillo or Boost, edit the [makefile](Makefile), which has variables for custom header and library search paths for these libraries (boost is header only).

# Example
A classic example for a stochastic dynamic optimization problem in economics is the neoclassical consumption model with **stochastic** income. Agent splits her income into consumption and savings and seeks the savings policy that maximizes her expected discounted utility from consumption over an infinite time horizon:

$\underset{k_t}{\text{max}} \ \mathbb{E} \left[\sum\limits_{t=0}^{\infty}\gamma^tU(c_t)\right]$

$\Leftrightarrow \underset{k_t}{\text{max}} \ \mathbb{E} \left[\sum\limits_{t=0}^{\infty}\gamma^tU(y_t - a(y_t))\right],$

s.t.

$0 \leq a(y) \leq y$, (feasibility constraint),

The transition function for income is

$y_{t+1} = a(y_t)^{\alpha}W_{t+1},$

with the action $k_t = a(y_t)$ representing the amount to save, given the income.

Popular choice for the shock is log-normal distribution $W \sim e^{N(0,1)}.$ For utility function, $U(c) = 1-e^{-\theta c}$.

More details:
- Stachurski, John. *Economic dynamics: theory and computation*. MIT Press, (2009).
- Stokey, Nancy, and R. Lucas. *Recursive Methods in Economic Dynamics* Harvard University Press (1989).

Implementation can be found here: [examples/optgrowth.cpp](examples/optgrowth.cpp).

## Solving the dynamic problem
Dynamic optimization problem such as the optimal consumption/savings can be solved with the help of the recursive [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation):

$V_t^{\pi}(S_t) \  =  \ \underset{\pi}{\text{max}} \ R_t(S_t,a_t) + \gamma \mathbb{E}\left[V_{t+1}^{\pi}(S_{t+1}|S_t)\right]$

The Bellman equation represents the value $V^{\pi}(s_t)$ of being in a state $s_t$ and following policy $\pi$.

For the optimal savings problem the Bellman equation represents the rewards/returns as

$R_t = U(s_t-a_t) + \gamma U(s_{t+1})$.

<!-- ## Discretizing the state and action variables -->
<!-- To discretize the state variable $y_t$, we go through these steps: -->

<!-- 1. Draw samples from the continuous distribution of state variable -->
<!-- 2. Divide the state space into bins and create discrete density(mass) function -->
<!-- 3. Create inverse cumulative distribution function form the discrete density function -->
<!-- 4. Use [inverse transform method](https://en.wikipedia.org/wiki/Inverse_transform_sampling) to sample from the resulting  discrete distribution -->

<!-- (Note that since we have the density function available for the state variable, instead of sampling we could use the density function directly to discretize the space.) -->

# C++ Implementation
Any model has to be derived from the base model struct (in [mc-control/model.hpp](mc-control/model.hpp)):

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


<!-- Let's discretize the state $y_t$ into 30 bins in the interval $[0.0,8.0]$ and action variable $k_t$ into 10 values. With 100k samples from $y_t$ the discrete approximation to the state-action density looks like this: -->

<!-- ![Discretized probability distribution](figures/discrete_density.png) -->

# The two implemented algorithms

1. Monte Carlo control with exploring starts ([Figure 5.4](figures/mc-es.png) in Sutton & Barto)
    - For infinite horizon problems (like the optimal savings problem), this algorithm reduces to randomly sampling the state-action space.
2. Monte Carlo control with a soft policy (epsilon greedy) ([Figure 5.6](figures/mc-soft-pol.png) in Sutton & Barto)

Both algorithms are implemented in file [mc-control/algorithms.hpp](mc-control/algorithms.hpp).

See chapter 5. in [*Reinforcement learning: An introduction*](http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html) for details.

#License

**mc-control** is made available under the terms of the GPLv3.

See the LICENSE file that accompanies this distribution for the full text of the license.
