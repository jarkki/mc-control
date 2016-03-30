/* Discrete probability distribution for Monte Carlo optimal control
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

#include <stdexcept>
#include <vector>
#include <tuple>
#include <armadillo>
#include "mc-control/utils.hpp"

using namespace std;
using namespace arma;
using namespace mc::utils;

namespace mc{

  namespace distributions{

    /*
      Creates a discrete distribution from a given sample (continuous or discrete) and bins.
      Allows one to draw samples from the resulting discretized distribution.
    */
    class DiscreteDistribution{
    public:

      // TODO: Constructor that constructs the discrete distribution from continuous density function

      DiscreteDistribution(mat samples, vector<vec> bins, vector<vec> bin_values){

        // TODO: Check that the the bins are equally spaced

        size_t nvariables = samples.n_cols;;
        size_t nsamples = samples.n_rows;

        // Initialize histograms to zero
        vector<vec> hists;
        for(auto variable : range(nvariables)){
          vec hist = arma::zeros(bins[variable].size()-1);
          hists.push_back(hist);
        }

        // Create histogram for each state variable
        for (auto sample : range(nsamples)){
          // For each state variable in this sample
          for(auto variable : range(nvariables)){
            // For each bin
            for (auto bin_i : range(bins[variable].size()-1)){
              auto state_value = samples(sample,variable);
              // If state variable value falls into this bin, increase histogram value for this bin
              if((bins[variable](bin_i) <= state_value) &&
                 (state_value < bins[variable](bin_i+1))){
                hists[variable](bin_i) += 1.0;
                break;
              }
            }
          }
        }

        // Create cumulative distributions and densities
        vector<vec> cumul_distrs;
        vector<vec> densities;
        vec bin_widths(nvariables);
        for(auto variable : range(nvariables)){

          // Normalize the histograms integrate to 1 (create densities)
          vec density = arma::zeros(bins[variable].size()-1);
          double dx = bins[variable](1) - bins[variable](0);
          density = hists[variable]/(arma::sum(hists[variable]) * dx);

          // Calculate cumulative distribution function
          vec cum_distr = arma::zeros(bins[variable].size());
          cum_distr(span(1,cum_distr.size()-1)) = arma::cumsum(density * dx);

          cumul_distrs.push_back(cum_distr);
          densities.push_back(density);
          bin_widths(variable) = dx;
        }

        // Calculate the number of bins for each variable
        vector<size_t> nbins(bins.size());
        for(auto variable : range(nvariables)){
          nbins[variable] = bins[variable].size();
        }

        this->nvariables = nvariables;
        this->cumul_distrs = cumul_distrs;
        this->nbins = nbins;
        this->bins = bins;
        this->bin_values = bin_values;
        this->bin_widths = bin_widths;
        this->densities = densities;
      }


      /*
        Inverse Uniform CDF sampling.

       */
      vector<size_t> sample() const{
        vector<size_t> state(this->nvariables);
        auto u = uniform(this->nvariables);
        for( auto variable : range(this->nvariables)){
          for(auto bin_i : range(this->nbins[variable])){
            if(u(variable) <= this->cumul_distrs[variable](bin_i+1)){
              state[variable] = bin_i;
              break;
            }
            // if((this->cumul_distrs[variable](bin_i) <= u(variable)) &&
            //    (u(variable) < this->cumul_distrs[variable](bin_i+1))){
            //   state[variable] = bin_i;
            //   break;
            // }
          }
        }
        return state;
      }

      size_t nvariables;
      vector<size_t> nbins;
      vector<vec> cumul_distrs;
      vector<vec> bins;
      vector<vec> bin_values;
      vec bin_widths;
      vector<vec> densities;
    };

  }

}
