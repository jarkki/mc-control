#pragma once

#include <math.h>
#include <armadillo>
#include "mc-control/utils.hpp"

using namespace std;
using namespace arma;
using namespace mc::utils;

namespace mc{

  namespace models{

    struct Model{

      virtual vec transition(const vec & state, const double & action) const = 0;
      virtual mat sample_transitions(const double & action, size_t n) const = 0;
      virtual double reward (const vec & state_value, const double & action_value, const vec & next_state_value) const = 0;
      virtual bool constraint(const double & action, const vec & state) const{
        return true;
      };

    };

  }

}
