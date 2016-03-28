#include <armadillo>
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

class OptimalGrowthModel : Model{
public:
  OptimalGrowthModel(){}
  OptimalGrowthModel(double theta_, double alpha_, double df_, mat state_lim_) : episode_length(1), state_lim(state_lim_),  nvariables(1), theta(theta_), alpha(alpha_), df(df_){};

  /*
    Sample from the model, given the action
  */
  mat sample(const double & action, size_t n=1) const{
    // Draw samples from log-norm distribution
    vec z  = arma::exp(mc::utils::norm(n));

    mat samples(n,1);

    for(auto i : range(n)){
      // y = k^alpha * z
      double y = std::pow(action,alpha) * z(i);
      samples(i,0) = y;
    }
    return samples;
  };

  /*
    Sample from the model, given the state and action.
    This model doesn't depend on the previous state, so the state is ignored.
  */
  mat sample(const vec & state, const double & action, int n=1) const {
    return this->sample(action,n);
  }

  bool constraint(const double & action, const vec & state) const {
    return ((this->state_lim[0] <= action) && (action <= state(0))) ? true : false;
  }

  double U(const double & c) const {
    return (1.0 - exp(- this->theta * c));
  }

  double reward (const vec & state_value, const double & action_value, const vec & next_state_value) const{
    return U(state_value(0) - action_value) + this->df * U(next_state_value(0));
  }

  size_t episode_length;
  mat state_lim;
  int nvariables;
  double theta;
  double alpha;
  double df;

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

  // State space
  mat state_lim (1,2);
  state_lim(0,0) = 0.0;
  state_lim(0,1) = 8.0;

  // Model parameters
  double theta = 0.5;
  double alpha = 0.8;
  double df = 0.9;

  // Bins for the discrete state space
  uvec nbins = {50};

  // Initialize the actions
  int nactions = 50;
  vec actions = linspace(state_lim(0), state_lim(1), nactions);

  // Instantiate the  model
  OptimalGrowthModel model(theta, alpha,  df, state_lim);

  // Instantiate decision problem
  OptimalGrowthProblem problem(model, actions, nbins, 10000);

  // // Plot the distributions
  // plot_distr(problem.distributions, problem.actions);

  // Run the MC-ES algorithm
  mat Q = run_mces(problem, 5000000);

  // Plot the Q-values
  plot_q(Q,problem);

  return 0;
}
