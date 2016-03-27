#pragma once

#include <math.h>
#include <armadillo>
#include "utils.hpp"

using namespace std;
using namespace arma;
using namespace mc::utils;

namespace mc{

  namespace models{

    class Model{

    };

    class OptimalGrowthModel{
    public:
      OptimalGrowthModel(){}
      OptimalGrowthModel(double theta_, double alpha_, double df_, mat state_lim_) : episode_length(1), state_lim(state_lim_),  nvariables(1), theta(theta_), alpha(alpha_), df(df_){


      };

      /*
        Sample from the model, given the action
       */
      mat sample(const double & action, size_t n=1) const{
        // Draw samples from log-norm distribution
        vec z  = arma::exp(mc::utils::norm(n));

        mat samples(n,1);

        for(auto i : range(n)){
          // y = k^alpha * z
          double y = std::pow(action,alpha) * z(i);
          samples(i,0) = y;
        }
        return samples;
      };

      /*
        Sample from the model, given the state and action.
        This model doesn't depend on the previous state, so the state is ignored.
      */
      mat sample(const vec & state, const double & action, int n=1) const {
        return this->sample(action,n);
      }

      bool constraint(const double & action, const vec & state) const {
        return ((this->state_lim[0] <= action) && (action <= state(0))) ? true : false;
      }

      double U(const double & c) const {
        return (1.0 - exp(- this->theta * c));
      }

      double reward (const vec & state_value, const double & action_value, const vec & next_state_value) const{
        return U(state_value(0) - action_value) + this->df * U(next_state_value(0));
      }

      size_t episode_length;
      mat state_lim;
      int nvariables;
      double theta;
      double alpha;
      double df;

    };

  }

}
