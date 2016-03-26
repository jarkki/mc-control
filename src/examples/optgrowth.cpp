#include <armadillo>
#include "utils.hpp"
#include "model.hpp"
#include "distribution.hpp"
#include "problem.hpp"
#include "algorithms.hpp"

using namespace std;
using namespace arma;
using namespace mc::utils;
using namespace mc::models;
using namespace mc::distributions;
using namespace mc::problem;
using namespace mc::algorithms;

int main(int argc, char *argv[])
{
  // State space
  mat state_lim (1,2);
  state_lim(0,0) = 0.0;
  state_lim(0,1) = 8.0;

  // Model parameters
  double theta = 0.5;
  double alpha = 0.8;
  double df = 0.9;

  // Bins for the discrete state space
  Col<int> nbins = {30};

  // Initialize the actions
  int nactions = 10;
  vec actions = linspace(state_lim(0), state_lim(1), nactions);

  // Instantiate the  model
  OptimalGrowthModel model(theta, alpha,  df, state_lim);

  // Instantiate decision problem
  DecisionProblem<OptimalGrowthModel> problem(model, actions, nbins, 10000);

  // Plot the distributions
  plot_distr(problem.distributions, problem.actions);

  // // Run the MC-ES algorithm
  // run_mces(problem, 100000);

  return 0;
}
