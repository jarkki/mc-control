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
     *  @param episode A function that completes one episode, given the starting state and action and following then the greedy policy. Defined as
     *
     *
     *    tuple<uvec,uvec,vec> episodes(const DiscretizedOptimalGrowthModel & discrete_model,
     *                                  const size_t & state,
     *                                  const size_t & action,
     *                                  const  uvec & pol);
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
      vector<uvec> possible_actions;

      size_t nstates = discrete_model.state_space_size;
      size_t nactions = discrete_model.nactions;

      // Init the matrices for Q-value, counter and returns
      mat Q = zeros(nstates,nactions);
      mat counter = zeros(nstates,nactions);
      mat returns = zeros(nstates,nactions);
      Mat<int> occurrences;

      // Init the possible actions matrix
      possible_actions = create_possible_actions_matrix(discrete_model);

      // Init random policy
      uvec pol = create_random_policy(possible_actions);

      // Main iteration loop
      for (auto iteration : range(niterations)){
        // Occurrences of state, action pairs in the episode (init to zero)
        occurrences = zeros<Mat<int> >(nstates,nactions);

        // Draw random starting state
        state = randint(nstates);
        state_value = discrete_model.state_values.row(state);

        // Select random action
        action = possible_actions[state](randint(possible_actions[state].size()));

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

        // Update policy to greedy policy
        for(auto state : episode_states){
          pol(state) = argmax_q(Q,state, possible_actions[state]);
          );
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
     *  @param discrete_model discretized model
     *  @param episode A function that completes one episode, following then soft policy. Defined as
     *
     *         tuple<uvec,uvec,vec> episodes(const DiscretizedOptimalGrowthModel & discrete_model,
     *                                       const  uvec & pol);
     *
     *
     *  @param niterations # of Monte Carlo iterations
     *  @param epsilon the probability for taking a soft(random) action (instead of greedy action)
     *
     *  @retval two-tuple of Q-value matrix and greedy policy vector
     *
     */
    template<typename DiscretizedModelT, typename EpisodeFuncT>
    tuple<mat,uvec> run_mc_eps_soft(const DiscretizedModelT & discrete_model,
                                           EpisodeFuncT episode,
                                           size_t niterations = 100000,
                                           double epsilon = 0.1){

          Mat<int> occurrences;
          uvec poss_actions, episode_states, episode_actions;
          vec state_value, qvals, episode_returns;
          tuple<uvec,uvec,vec> episode_result;
          vector<uvec> possible_actions;

          size_t nstates = discrete_model.state_space_size;
          size_t nactions = discrete_model.nactions;

          // Init the matrices for Q-value, counter and returns
          mat Q = zeros(nstates,nactions);
          mat counter = zeros(nstates,nactions);
          mat returns = zeros(nstates,nactions);

          // Init the possible actions matrix
          possible_actions = create_possible_actions_matrix(discrete_model);

          // Init random policy
          uvec pol = create_random_policy(possible_actions);

          // Main iteration loop
          for (auto iteration : range(niterations)){

            // Occurrences of state, action pairs in the episode (init to zero)
            occurrences = zeros<Mat<int> >(nstates,nactions);

            // Generate episode using the epsilon-soft policy
            tie(episode_states, episode_actions, episode_returns) = episode(discrete_model, pol);

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

            // Update policy with epsilon-greedy selection
            for(auto state : episode_states){

              if(uniform() < epsilon){
                //Random action
                pol(state) = possible_actions[state](randint(possible_actions[state].size()));
              }else{
                // Greedy action for policy
                pol(state) = argmax_q(Q, state, possible_actions[state]);
              }
            }

            // Print info
            if(iteration % 10000 == 0 && iteration > 0){
              cout << "Iteration " << iteration << endl;
            }

          }

          // Calculate greedy policy
          for(auto state : range(nstates)){
            pol(state) = argmax_q(Q, state, possible_actions[state]);
          }

          return make_tuple(Q, pol);
        }

  }
}
