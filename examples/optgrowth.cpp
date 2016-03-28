#include <armadillo>
#include <math.h>
#include "mc-control/utils.hpp"
#include "mc-control/model.hpp"
#include "mc-control/distribution.hpp"
#include "mc-control/problem.hpp"
#include "mc-control/algorithms.hpp"
#include "mc-control/plot.hpp"

using namespace std;
using namespace arma;
using namespace mc::utils;
using namespace mc::models;
using namespace mc::distributions;
using namespace mc::problem;
using namespace mc::algorithms;
using namespace mc::plot;

/*
  Transition function

  y = k^alpha * z,

  where k is the action (amount to save) and z is lognormal shock
*/
struct OptimalGrowthModel : Model{

  size_t nvariables;
  double theta; // Utility func parameter
  double alpha; // Transition func parameter
  double df;    // Discount factor
  mat state_lim;// Limits of state space

  OptimalGrowthModel(){} // Default constructor
  OptimalGrowthModel(mat state_lim, double theta = 0.5, double alpha = 0.8, double df = 0.9){
    this->nvariables = 1;
    this->theta = theta;
    this->alpha = alpha;
    this->df = df;
    this->state_lim = state_lim;
  }

  /*
    Transition function doesn't depend on state, only on action
    y = k^alpha * z
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

class OptimalGrowthProblem : public DecisionProblem<OptimalGrowthModel>{
public:
  OptimalGrowthProblem(const OptimalGrowthModel &  model, const vec & actions,  uvec nbins, int nsamples) : DecisionProblem<OptimalGrowthModel>(model,actions,nbins,nsamples){};

  /*
    Complete one episode of the problem.
    This is infinite horizon problem, so the length of the episode is 1 and the policy is not used.
  */
  tuple<uvec,uvec,vec> episode( size_t  state,  size_t action, const  uvec & pol) const{

    uvec states(1);
    uvec actions(1);
    vec rewards(1);
    size_t next_state;
    vector<size_t> std_state_vec;

    states(0) = state;
    actions(0) = action;

    // Sample next state
    std_state_vec = this->distributions[action].sample();

    // Get the state index from the state-index map
    next_state = this->state_index_map.at(std_state_vec);

    // Calculate reward for being in state, taking action and ending in next_state
    rewards(0) = this->model.reward(this->state_values.row(state), this->actions(action), this->state_values.row(next_state));

    return make_tuple(states,actions,rewards);
  }

};


int main(int argc, char *argv[])
{
  // Seed the rng randomly
  arma_rng::set_seed_random();

  // State space limits
  mat state_lim = {{0.0, 8.0},};

  // Instantiate the  model
  OptimalGrowthModel model(state_lim);

  // Number of bins for the discrete state space
  uvec nbins = {30};

  // Initialize the actions
  int nactions = 20;
  vec actions = linspace(state_lim(0,0), state_lim(0,1), nactions);

  // Instantiate decision problem
  OptimalGrowthProblem problem(model, actions, nbins, 10000);

  // // Plot the distributions
  // plot_distr(problem.distributions, problem.actions);

  // Run the MC-ES algorithm
  mat Q;
  uvec pol;
  tie(Q,pol) = run_mc_es(problem, 1000000);

  // Plot the Q-values
  plot_q(Q,pol,problem);

  return 0;
}
