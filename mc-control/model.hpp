#pragma once

#include <math.h>
#include <armadillo>
#include "mc-control/utils.hpp"

using namespace std;
using namespace arma;
using namespace mc::utils;

namespace mc{

  namespace models{

    class Model{
      virtual mat sample(const double & action, size_t n=1) const = 0;
      virtual double reward (const vec & state_value, const double & action_value, const vec & next_state_value) const = 0;
      virtual bool constraint(const double & action, const vec & state) const{
        return true;
      };

    };

  }

}
