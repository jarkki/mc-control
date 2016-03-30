#include <armadillo>
#include <math.h>
#include "mc-control/utils.hpp"
#include "mc-control/model.hpp"
#include "mc-control/distribution.hpp"
#include "mc-control/algorithms.hpp"
#include "mc-control/plot.hpp"

using namespace std;
using namespace arma;
using namespace mc::utils;
using namespace mc::models;
using namespace mc::algorithms;
using namespace mc::plot;


/*! Optimal Growth model
 *
 *
 *  This model represents the classic
 *
 *  @param param
 *
 *  @retval return type
 */
struct OptimalGrowthModel : Model{

  size_t nvariables;
  double theta; // Utility func parameter
  double alpha; // Transition func parameter
  double df;    // Discount factor
  mat state_lim;// Limits of state space

  //! Default constructor
  OptimalGrowthModel(){}

  /*! Constructor
   *
   *
   *  Detailed description
   *
   *  @param param
   *
   *  @retval return type
   */

  OptimalGrowthModel(mat state_lim, double theta = 0.5, double alpha = 0.8, double df = 0.9){
    this->nvariables = 1;
    this->theta = theta;
    this->alpha = alpha;
    this->df = df;
    this->state_lim = state_lim;
  }

  /*
    Transition function doesn't depend on state, only on action:
      y = k^alpha * z,
    where k is the action.
   */
  vec transition(const vec & state, const double & action) const{
    // Draw a sample from log-norm distribution
    vec next_state(1);
    next_state(0) = std::pow(action,this->alpha) * std::exp(mc::utils::norm());
    return next_state;
  };

  /*
    Create a sample of transitions, given the action.
   */
  mat sample_transitions(const double & action, size_t n) const{

    mat samples(n,1);
    vec state(1); // Doesn't depend on state, but have to pass state anyway
    for(auto i : range(n)){
      samples(i,0) = this->transition(state, action)(0);
    }
    return samples;
  }

  /*
    Returns true if it is possible to take the action from this state.
  */
  bool constraint(const double & action, const vec & state) const {
    return ((this->state_lim[0] <= action) && (action <= state(0))) ? true : false;
  }

  /*
    Reward for being in state, taking action and ending in next_state
  */
  double reward (const vec & state_value, const double & action_value, const vec & next_state_value) const{
    return U(state_value(0) - action_value) + this->df * U(next_state_value(0));
  }

  /*
    Utility function.
  */
  double U(const double & c) const {
    return (1.0 - exp(- this->theta * c));
  }


};

// Typedef for clearer (at least somewhat) code
typedef DiscretizedModel<OptimalGrowthModel> DiscretizedOptimalGrowthModel;
typedef SoftPolicy<DiscretizedModel<OptimalGrowthModel> > OptimalGrowthSoftPolicy;


/*! Simulate one episode from the optimal growth model WITH EXPLORING STARTS.
 *
 *
 *  This function simulates one episode. note that this is only for the Monte Carlo control algorithm WITH EXPLORING STARTS.
 *
 *  @param discrete_model : The discretized model
 *  @param state          : The state where to start from
 *  @param action         : The randomly selected action to start with
 *  @param pol            : The policy function policy(state)
  *
 *  @retval Tuple with all states, actions and returns that happened during the episode.
 */
tuple<uvec,uvec,vec> episode_es(const DiscretizedOptimalGrowthModel & discrete_model,  const size_t & state,  const size_t & action, const  uvec & pol) {

  uvec states(1);
  uvec actions(1);
  vec returns(1);
  size_t next_state;
  vector<size_t> std_state_vec;

  states(0) = state;
  actions(0) = action;

  // Sample next state
  std_state_vec = discrete_model.distributions[action].sample();

  // Get the state index from the state-index map
  next_state = discrete_model.state_index_map.at(std_state_vec);

  // Calculate reward for being in state, taking action and ending in next_state
  returns(0) = discrete_model.model.reward(discrete_model.state_values.row(state), discrete_model.actions(action), discrete_model.state_values.row(next_state));

  return make_tuple(states,actions,returns);
}


/*! Simulate one episode from the optimal growth model WITH SOFT EPSILON POLICIES.
 *
 *
 *  This function simulates one episode. Note that this is only for the Monte Carlo control algorithm WITH SOFT EPSILON POLICIES.
 *
 *  @param discrete_model : The discretized model
 *  @param start_state    : Starting state
 *  @param soft_pol       : Soft policy.
 *
 *  @retval Tuple with all states, actions and returns that happened during the episode.
 */
tuple<uvec,uvec,vec> episode_soft_pol(const DiscretizedOptimalGrowthModel & discrete_model, size_t start_state, const OptimalGrowthSoftPolicy & soft_pol) {

  uvec states(1);
  uvec actions(1);
  vec returns(1);
  size_t state, action, next_state;
  vector<size_t> std_state_vec;

  state = start_state;

  // Sample action from the soft policy density for this state
  action = soft_pol[state].sample();

  // Sample next state
  std_state_vec = discrete_model.distributions[action].sample();

  // Get the state index from the state-index map
  next_state = discrete_model.state_index_map.at(std_state_vec);

  // Calculate reward for being in state, taking action and ending in next_state
  returns(0) = discrete_model.model.reward(discrete_model.state_values.row(state), discrete_model.actions(action), discrete_model.state_values.row(next_state));

  states(0) = state;
  actions(0) = action;

  return make_tuple(states,actions,returns);
}


int main(int argc, char *argv[])
{
  // Seed the rng (randomly)
  arma_rng::set_seed_random();

  // State space limits
  mat state_lim = {{0.0, 8.0},};

  // Instantiate the  model
  OptimalGrowthModel model(state_lim);

  // Number of bins for the discrete state space
  uvec nbins = {30};

  // Initialize the actions
  int nactions = 10;
  //vec actions = linspace(state_lim(0,0), state_lim(0,1), nactions);
  vec actions = linspace(0.5, state_lim(0,1), nactions);

  // Create discretized model from the model
  DiscretizedModel<OptimalGrowthModel> discrete_model(model, actions, nbins, 100000);

  // Plot the distributions
  plot_distr(discrete_model.distributions, discrete_model.actions);

  // // Run the MC-ES algorithm
  // mat Q;
  // uvec pol;
  // tie(Q,pol) = run_mc_es(discrete_model, episode_es, 30000000);
  // // tie(Q,pol) = run_mc_eps_soft(discrete_model, episode_soft_pol, 30000000, 0.7);

  // // Plot the Q-values
  // plot_q(Q,pol,discrete_model);


  return 0;
}
