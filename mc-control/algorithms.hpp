#pragma once

#include <stdexcept>
#include <vector>
#include <map>
#include <tuple>
#include <armadillo>
#include "mc-control/utils.hpp"
#include "mc-control/distribution.hpp"
#include "mc-control/problem.hpp"

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


    int argmax_q(const mat & Q, const int &state, const uvec & possible_a){

      // There might be several actions that give the max Q-value
      vector<size_t> maxq_actions;

      // Use a subset of actions
      double maxq = Q(state, possible_a(0));
      for(auto i : range(possible_a.size())){
        if(Q(state,possible_a(i)) > maxq){
          maxq = Q(state,possible_a(i));
        }
      }
      // Check if there are several actions with same q-value as maxq
      for(auto i : range(possible_a.size())){
        if (Q(state,possible_a(i)) == maxq){
          maxq_actions.push_back(possible_a(i));
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
    tuple<mat,uvec> run_mces(const ProblemT & problem,
                             size_t niterations = 100000){

      size_t nstates = problem.state_space_size;
      size_t nactions = problem.nactions;

      // Init the matrices for Q-value, counter and rewards
      mat Q = zeros(nstates,nactions);
      mat counter = zeros(nstates,nactions);
      mat rewards = zeros(nstates,nactions);
      Mat<int> occurrences;

      // Init random policy
      uvec pol(nstates);
      for(auto i : range(nstates)){
        uvec poss = possible_actions(problem.state_values.row(i), problem.actions, problem.model);
        pol(i) = poss(randint(poss.size()));
      }

      uvec poss_actions, episode_states, episode_actions;
      int state, action;
      vec state_value, qvals, episode_rewards;
      tuple<uvec,uvec,vec> episode_result;

      for (auto iteration : range(niterations)){
        // Occurrences of state, action pairs in the episode (init to zero)
        occurrences = zeros<Mat<int> >(nstates,nactions);

        // Draw random starting state
        state = randint(nstates);
        state_value = problem.state_values.row(state);

        // Select random action
        poss_actions = possible_actions(state_value, problem.actions, problem.model);
        action = poss_actions(randint(poss_actions.size()));

        // Run episode, starting from state, action and then following policy pol
        episode_result = problem.episode(state, action, pol);
        tie(episode_states, episode_actions, episode_rewards) = episode_result;

        // For each state, action pair in episode
        for(auto i : range(episode_states.size())){
          size_t s = episode_states(i);
          size_t a = episode_actions(i);
          // If this is first occurrence of state, action
          if(occurrences(s,a) == 0){
            // Append to rewards
            rewards(s,a) += episode_rewards(i);
            // Increase counter
            counter(s,a) += 1;
            // Update Q-value
            Q(s,a) = rewards(s,a)/counter(s,a);
            // Mark this state, action pair as occurred
            occurrences(s,a) = 1;
          }
        }

        // Update policy
        for(auto state : episode_states){
          uvec poss_a = possible_actions(problem.state_values.row(state), problem.actions, problem.model);
          pol(state) = argmax_q(Q,state,poss_a);
        }
        // Print info
        if(iteration % 10000 == 0){
          cout << "Iteration " << iteration << endl;
        }

      }
      return make_tuple(Q, pol);
    }
  }
}
