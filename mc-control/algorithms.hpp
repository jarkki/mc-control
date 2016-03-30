#pragma once

#include <stdexcept>
#include <vector>
#include <utility>
#include <map>
#include <tuple>
#include <armadillo>
#include "mc-control/utils.hpp"
#include "mc-control/distribution.hpp"
#include "mc-control/model.hpp"

using namespace std;
using namespace arma;
using namespace mc::utils;
using namespace mc::distributions;
using namespace mc::models;

namespace mc{

  namespace algorithms{


    /*! Monte Carlo control with exploring starts.
     *
     *
     *  Starts each episode with a randomly selected state and action.
     *
     *  For an infinite horizon problem (episode length = 1), this reduces to just randomly sampling the state-action space.
     *
     *  @param discrete_model discretized model
     *  @param episode A function that completes one episode, given the starting state and action and following a given policy. Defined as 
     *
     *
     *  tuple<uvec,uvec,vec> episodes(const DiscretizedOptimalGrowthModel & discrete_model,  const size_t & state,  const size_t & action, const  uvec & pol);
     *
     *
     *  @param niterations # of Monte Carlo iterations
     *
     *  @retval two-tuple of Q-value matrix and greedy policy vector
     */
    template<typename DiscretizedModelT, typename EpisodeFuncT>
    tuple<mat,uvec> run_mc_es(const DiscretizedModelT & discrete_model,
                              EpisodeFuncT episode,
                              size_t niterations = 100000){

      uvec poss_actions, episode_states, episode_actions;
      size_t state, action;
      vec state_value, qvals, episode_returns;
      tuple<uvec,uvec,vec> episode_result;

      size_t nstates = discrete_model.state_space_size;
      size_t nactions = discrete_model.nactions;

      // Init the matrices for Q-value, counter and returns
      mat Q = zeros(nstates,nactions);
      mat counter = zeros(nstates,nactions);
      mat returns = zeros(nstates,nactions);
      Mat<int> occurrences;

      // Init random policy
      uvec pol(nstates);
      for(auto i : range(nstates)){
        uvec poss = possible_actions(discrete_model.state_values.row(i), discrete_model.actions, discrete_model.model);
        pol(i) = poss(randint(poss.size()));
      }

      // Main iteration loop
      for (auto iteration : range(niterations)){
        // Occurrences of state, action pairs in the episode (init to zero)
        occurrences = zeros<Mat<int> >(nstates,nactions);

        // Draw random starting state
        state = randint(nstates);
        state_value = discrete_model.state_values.row(state);

        // Select random action
        poss_actions = possible_actions(state_value, discrete_model.actions, discrete_model.model);
        action = poss_actions(randint(poss_actions.size()));

        // Run episode, starting from state, action and then following policy pol
        tie(episode_states, episode_actions, episode_returns) = episode(discrete_model, state, action, pol);

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

        // Update policy to the greedy policy
        for(auto state : episode_states){
          uvec poss_a = possible_actions(discrete_model.state_values.row(state), discrete_model.actions, discrete_model.model);
          pol(state) = argmax_q(Q,state,poss_a);
        }

        // Print info
        if(iteration % 10000 == 0 && iteration > 0){
          cout << "Iteration " << iteration << endl;
        }

      }
      return make_tuple(Q, pol);
    }


    /*! Monte Carlo control with epsilon-soft policies.
     *
     *
     *  
     *
     *  @param epsilon the probability for taking a soft action (instead of greedy action)
     *
     *  @retval return type
     */
    template<typename DiscretizedModelT, typename EpisodeFuncT>
    tuple<mat,uvec> run_mc_eps_soft(const DiscretizedModelT & discrete_model,
                                    EpisodeFuncT episode,
                                    size_t niterations = 100000,
                                    double epsilon = 0.1){

      Mat<int> occurrences;
      uvec poss_actions, episode_states, episode_actions;
      size_t state;
      vec state_value, qvals, episode_returns;
      tuple<uvec,uvec,vec> episode_result;

      size_t nstates = discrete_model.state_space_size;
      size_t nactions = discrete_model.nactions;

      // Init the matrices for Q-value, counter and returns
      mat Q = zeros(nstates,nactions);
      mat counter = zeros(nstates,nactions);
      mat returns = zeros(nstates,nactions);

      // Init soft policy for each state, initialize to equal probabilities
      SoftPolicy<DiscretizedModelT> soft_pol(discrete_model);

      // Init the greedy policy
      uvec pol(nstates);

      // Main iteration loop
      for (auto iteration : range(niterations)){

        // Occurrences of state, action pairs in the episode (init to zero)
        occurrences = zeros<Mat<int> >(nstates,nactions);

        // Draw random starting state
        state = randint(nstates);
        state_value = discrete_model.state_values.row(state);

        // Generate episode using the epsilon-soft policy
        tie(episode_states, episode_actions, episode_returns) = episode(discrete_model, state, soft_pol);

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

        // Update policy for each state in episode
        for(auto state : episode_states){

          // Greedy action
          size_t greedy_action = argmax_q(Q, state, soft_pol[state].actions);

          // For each action possible from this state
          size_t nactions_s = soft_pol[state].actions.size();
          for(auto i : range(nactions_s)){

            // Greedy action gets the highest probability and other actions share low probability
            if (soft_pol[state].actions(i) == greedy_action){
              soft_pol[state].density(i) = 1.0 - epsilon + (epsilon/static_cast<double>(nactions_s));
            }else{
              soft_pol[state].density(i) = epsilon/static_cast<double>(nactions_s);
            }
          }
        }

        // Print info
        if(iteration % 10000 == 0 && iteration > 0){
          cout << "Iteration " << iteration << endl;
        }

      }

      // Calculate greedy policy
      for(auto state : range(nstates)){
        pol(state) = argmax_q(Q, state, soft_pol[state].actions);
      }

      return make_tuple(Q, pol);
    }

  }
}
