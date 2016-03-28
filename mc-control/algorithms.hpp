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

    /*
     *
     * Use inverse uniform cdf sampling to sample from discrete distribution
     *
     */
    size_t sample_discrete(const vec & densities){
      size_t n = densities.size();
      vec cum_prob = zeros(n+1);
      cum_prob(span(1, n))= cumsum(densities);
      double u = uniform();
      for(auto i : range(n)){
        if(u <= cum_prob(i+1)){
          return i;
        }
      }
      throw invalid_argument("Function sample_discrete failed...");
    }

    /*
     *
     * Returns the possible actions for a given state.
     *
     */
    template<typename ModelT>
    uvec possible_actions(const vec & state, const vec & actions, const ModelT & model){

      size_t nactions = actions.size();
      vector<size_t> possible;
      for(auto action : range(nactions)){
        if(model.constraint(actions(action), state)){
          possible.push_back(action);
        }
      }
      uvec poss = conv_to<uvec>::from(possible);
      return poss;
    };

    /*
     *
     * Returns the action that maximizes the Q-value for the given state
     *
     */
    size_t argmax_q(const mat & Q, const size_t &state, const uvec & possible_a){

      // There might be several actions that give the max Q-value
      vector<size_t> maxq_actions;

      // Calculate the max Q-value for this state and actions possible from here
      double maxq = Q(state, possible_a(0));
      for(auto i : range(1,possible_a.size())){
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

      // Return the single action that maximizes the Q-value or
      //  randomly one of the actions that maximize the Q-value.
      if(maxq_actions.size() > 1){
        return maxq_actions[randint(maxq_actions.size())];
      }else{
        return maxq_actions[0];
      }
    }

    /*

      Monte Carlo control with exploring starts.

    */
    template<typename ProblemT>
    tuple<mat,uvec> run_mc_es(const ProblemT & problem,
                             size_t niterations = 100000){

      size_t nstates = problem.state_space_size;
      size_t nactions = problem.nactions;

      // Init the matrices for Q-value, counter and returns
      mat Q = zeros(nstates,nactions);
      mat counter = zeros(nstates,nactions);
      mat returns = zeros(nstates,nactions);
      Mat<int> occurrences;

      // Init random policy
      uvec pol(nstates);
      for(auto i : range(nstates)){
        uvec poss = possible_actions(problem.state_values.row(i), problem.actions, problem.model);
        pol(i) = poss(randint(poss.size()));
      }

      uvec poss_actions, episode_states, episode_actions;
      size_t state, action;
      vec state_value, qvals, episode_returns;
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
        tie(episode_states, episode_actions, episode_returns) = episode_result;

        // For each state, action pair in episode
        for(auto i : range(episode_states.size())){
          size_t s = episode_states(i);
          size_t a = episode_actions(i);
          // If this is first occurrence of state, action
          if(occurrences(s,a) == 0){
            // Append to returns
            returns(s,a) += episode_returns(i);
            // Increase counter
            counter(s,a) += 1;
            // Update Q-value
            Q(s,a) = returns(s,a)/counter(s,a);
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
        if(iteration % 10000 == 0 && iteration > 0){
          cout << "Iteration " << iteration << endl;
        }

      }
      return make_tuple(Q, pol);
    }


    /*

      Monte Carlo control with epsilon-soft policies. No exploring starts.

    */
    template<typename ProblemT>
    tuple<mat,uvec> run_mc_eps_soft(const ProblemT & problem,
                                 size_t niterations = 100000){

      size_t nstates = problem.state_space_size;
      size_t nactions = problem.nactions;

      // Init the matrices for Q-value, counter and returns
      mat Q = zeros(nstates,nactions);
      mat counter = zeros(nstates,nactions);
      mat returns = zeros(nstates,nactions);
      Mat<int> occurrences;

      // Init random policy
      uvec pol(nstates);
      for(auto i : range(nstates)){
        uvec poss = possible_actions(problem.state_values.row(i), problem.actions, problem.model);
        pol(i) = poss(randint(poss.size()));
      }

      uvec poss_actions, episode_states, episode_actions;
      size_t state, action;
      vec state_value, qvals, episode_returns;
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
        tie(episode_states, episode_actions, episode_returns) = episode_result;

        // For each state, action pair in episode
        for(auto i : range(episode_states.size())){
          size_t s = episode_states(i);
          size_t a = episode_actions(i);
          // If this is first occurrence of state, action
          if(occurrences(s,a) == 0){
            // Append to returns
            returns(s,a) += episode_returns(i);
            // Increase counter
            counter(s,a) += 1;
            // Update Q-value
            Q(s,a) = returns(s,a)/counter(s,a);
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
