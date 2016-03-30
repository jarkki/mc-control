/* Model base class and discretized model for Monte Carlo optimal control
 *
 * Copyright (C) 2016  Jarno Kiviaho <jarkki@kapsi.fi>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#pragma once

#include <vector>
#include <map>
#include <math.h>
#include <armadillo>
#include "mc-control/utils.hpp"
#include "mc-control/distribution.hpp"

using namespace std;
using namespace arma;
using namespace mc::utils;
using namespace mc::distributions;

namespace mc{

  namespace models{

    /*! Abstract base class for the models
     *
     */
    struct Model{

      /*! next_state = f(state, action)*/
      virtual vec transition(const vec & state, const double & action) const = 0;

      /*! Samples the transition funciton n times*/
      virtual mat sample_transitions(const double & action, size_t n) const = 0;

      /*! Reward from being in a state, taking action and ending in next_state */
      virtual double reward (const vec & state_value, const double & action_value, const vec & next_state_value) const = 0;

      /*! Returns true if it is possible to take the action from this state */
      virtual bool constraint(const double & action, const vec & state) const{
        return true;
      };

    };

    /*! Creates a discretized version of a given continuous state model
     *
     *
     *  Discretizes the state space and constructs an inverse cumulative distribution
     *   function for inverse transform sampling.
     *
     */
    template <typename ModelT>
    class DiscretizedModel{
    public:

      /*! Constructor
       *
       *  \param model    : continuous state model derived from the abstract model base class
       *  \param actions  : vector of discrete points in continuous action space
       *  \param nbins    : vector, # of bins for each variable
       *  \param nsamples : # of the samples to draw from the model for the discretization
       *
       */
      DiscretizedModel(const ModelT &  model, const vec & actions,  uvec nbins, int nsamples){
        size_t nactions = actions.size();

        // Create bins for discretization of each state and
        //  calculate middle values for the bins.
        vector<vec> bins;
        vector<vec> bin_values;
        vec bin_widths(model.nvariables);
        for (auto state : range(model.nvariables)){
          vec state_bins = linspace(model.state_lim(state,0), model.state_lim(state,1), nbins(state)+1);
          vec values(nbins(state));
          for(auto bin_i : range(nbins(state))){
            values(bin_i) = (state_bins(bin_i) + state_bins(bin_i+1))/2.0;
          }
          bins.push_back(state_bins);
          bin_values.push_back(values);
          bin_widths(state) = state_bins(1) - state_bins(0);
        }

        // Discretize the model from a sample
        // Create distribution for each action
        vector<DiscreteDistribution> distributions;
        for(auto action : actions){
          // Sample the transition function with this action
          auto sample = model.sample_transitions(action, nsamples);
          DiscreteDistribution distr(sample, bins, bin_values);
          distributions.push_back(distr);
        }

        // Index the state space with all possible combinations of state variables
        Mat<size_t> state_space = combinations(nbins);

        // Index the state values
        mat state_values(size(state_space));
        size_t state_space_size = state_space.n_rows;
        for(auto state_i : range(state_space_size)){
          for(auto var_i : range(model.nvariables)){
            state_values(state_i,var_i) = bin_values[var_i](state_i);
          }
        }

        // Create reverse index on state space with std::map, so after sampling state variables
        //   you can find the index of the state variables. 
        // Use like map[statevec] = state_index
        // (std::vector can be used as a key in a map, arma::vec cannot)
        map<vector<size_t>, size_t> state_index_map;
        for(auto i : range(state_space_size)){
          vector<size_t> std_state  = conv_to<vector<size_t> >::from(state_space.row(i));
          state_index_map[std_state] = i;
        }

        this->model = model;
        this->distributions = distributions;
        this->actions = actions;
        this->nactions = nactions;
        this->bins = bins;
        this->bin_widths = bin_widths;
        this->bin_values = bin_values;
        this->state_space = state_space;
        this->state_values = state_values;
        this->state_space_size = state_space_size;
        this->state_index_map = state_index_map;
      }

      ModelT model;
      vector<DiscreteDistribution> distributions;
      vec actions;
      size_t nactions;
      vector<vec> bins;
      vector<vec> bin_values;
      vec bin_widths;
      Mat<size_t> state_space;
      mat state_values;
      size_t state_space_size;
      map<vector<size_t>, size_t> state_index_map;
    };


  }

}
