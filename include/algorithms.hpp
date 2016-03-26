#pragma once

#include <stdexcept>
#include <vector>
#include <tuple>
#include <boost/range/irange.hpp>
#include <armadillo>
#include "utils.hpp"
#include "distribution.hpp"
#include "problem.hpp"

using namespace std;
using namespace arma;
using namespace mc::utils;
using namespace mc::distributions;
using namespace mc::problem;

namespace mc{

  namespace algorithms{


    template<typename ModelT>
    Col<int> possible_actions(const vec & state, const vec & actions, const ModelT & model){

      int nactions = actions.size();
      vector<int> possible;
      for(int action : range(nactions)){
        if(model.constraint(actions(action), state)){
          possible.push_back(action);
        }
      }
      Col<int> poss = conv_to<Col<int> >::from(possible);
      return poss;
    };

    /*
      Monte Carlo control with exploring starts and epsilon greedy action selection
     */
    template<typename ProblemT>
    mat run_mces(const ProblemT & problem,
                 int niterations = 100000,
                 double epsilon = 0.1){

      int nstates = problem.model.nstates;
      int nactions = problem.nactions;

      // Init the matrices for Q-value, counter and rewards
      mat Q = zeros(nstates,nactions);
      mat counter = zeros(nstates,nactions);
      mat rewards = zeros(nstates,nactions);

      Col<int> poss_actions;
      int state_var, next_state, action;
      vec state_value;

      for (int iteration : range(niterations)){

        // // Draw random starting state
        // state_var = randint(nstates);
        // //state_value = problem.bin_values[state];

        // // Epsilon-greedy action selection
        // poss_actions = possible_actions(state_value, problem.actions, problem.model);
        // state_value.print();
        // poss_actions.print();

        break;

        // Print info
        if(iteration % 10000 == 0){
          cout << "Iteration " << iteration << endl;
        }

      }
      return Q;
    }
  }
}
