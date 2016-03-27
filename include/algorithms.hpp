#pragma once

#include <stdexcept>
#include <vector>
#include <tuple>
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
    uvec possible_actions(const vec & state, const vec & actions, const ModelT & model){

      size_t nactions = actions.size();
      vector<int> possible;
      for(auto action : range(nactions)){
        if(model.constraint(actions(action), state)){
          possible.push_back(action);
        }
      }
      uvec poss = conv_to<uvec>::from(possible);
      return poss;
    };


    int argmax_q(const mat & Q, const int &state, const uvec & actions){
      // Calculate the max Q(state)
      double maxq = max(Q.row(state));
      // Check if there are several actions with same q-value as maxq
      size_t nactions = actions.size();
      vector<size_t> maxq_actions;
      for(auto a : range(nactions)){
        if (Q(state,a) == maxq){
          maxq_actions.push_back(a);
        }
      }
      if(maxq_actions.size() > 1){
        // Return randomly one of the actions that maximize the Q-value
        return maxq_actions[randint(maxq_actions.size())];
      }else{
        // Return the single action that maximizes the Q-value
        return maxq_actions[0];
      }
    }

    /*
      Monte Carlo control with exploring starts and epsilon greedy action selection
     */
    template<typename ProblemT>
    mat run_mces(const ProblemT & problem,
                 size_t niterations = 100000){

      size_t nstates = problem.state_space_size;
      size_t nactions = problem.nactions;

      // Init the matrices for Q-value, counter and rewards
      mat Q = zeros(nstates,nactions);
      mat counter = zeros(nstates,nactions);
      mat rewards = zeros(nstates,nactions);

      // Init random policy
      uvec pol(nstates);
      for(auto i : range(nstates)){
        uvec poss = possible_actions(problem.state_values.row(i), problem.actions, problem.model);
        int poss_len = poss.size();
        pol(i) = poss(randint(poss_len));
      }
      cout << "Pol: " << endl;
      pol.print();

      uvec poss_actions;
      int state, next_state, action, poss_len;
      vec state_value, qvals;

      for (auto iteration : range(niterations)){

        // Draw random starting state
        state = randint(nstates);
        state_value = problem.state_values.row(state);

        // Select random action
        poss_actions = possible_actions(state_value, problem.actions, problem.model);
        poss_len = poss_actions.size();
        action = poss_actions(randint(poss_actions.size()));

        cout << "State " << state << endl;
        cout << "Action " << action << endl;
        poss_actions.print();

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
