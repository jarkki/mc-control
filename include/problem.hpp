#pragma once

#include <stdexcept>
#include <vector>
#include <tuple>
#include <boost/range/irange.hpp>
#include <armadillo>
#include "utils.hpp"
#include "distribution.hpp"

using namespace std;
using namespace arma;
using namespace mc::utils;
using namespace mc::distributions;

namespace mc{

  namespace problem{

    template <typename ModelT>
    class DecisionProblem{
    public:
      DecisionProblem(const ModelT &  model, const vec & actions,  Col<int> nbins, int nsamples){
        int nactions = actions.size();

        // Create bins for discretization of each state and
        //  calculate middle values for the bins.
        vector<vec> bins;
        vector<vec> bin_values;
        vec bin_widths(model.nstates);
        for (int state : range(model.nstates)){
          vec state_bins = linspace(model.state_lim(state,0), model.state_lim(state,1), nbins(state)+1);
          vec values(nbins(state));
          for(int bin_i : range(nbins(state))){
            values(bin_i) = (state_bins(bin_i) + state_bins(bin_i+1))/2.0;
          }
          bins.push_back(state_bins);
          bin_values.push_back(values);
          bin_widths(state) = state_bins(1) - state_bins(0);
        }

        // Discretize the model from a sample
        // Create distribution for each action
        vector<DiscreteDistribution> distributions;
        for(int action : range(nactions)){
          cout << "Discretizing action " << action << "..." << endl;
          // Sample for this action
          auto sample = model.sample(actions(action), nsamples);
          DiscreteDistribution distr(sample, bins, bin_values);
          distributions.push_back(distr);
        }

        // Index the state space
        Mat<int> state_space = combinations(nbins);
        mat state_values(size(state_space));
        int state_space_size = state_space.n_rows;
        for(int i : range(state_space_size)){
          for(int var_i : range(model.nstates)){
            state_values(i,var_i) = bin_values[var_i](i);
          }
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
      }

      ModelT model;
      vector<DiscreteDistribution> distributions;
      vec actions;
      int nactions;
      vector<vec> bins;
      vector<vec> bin_values;
      vec bin_widths;
      Mat<int> state_space;
      mat state_values;
      int state_space_size;
    };

  }
}
