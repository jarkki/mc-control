#include "distribution.hpp"

using namespace std;
using namespace arma;
using namespace mc::utils;

namespace mc{

  namespace distribution{

    DiscreteDistribution::DiscreteDistribution(mat samples, vector<vec> bins){
      size_t nstates = samples.n_cols;
      size_t nbins = bins.n_rows - 1;

      // Create histogram
      vec hist = arma::zeros(nbins,nstates);
      for(size_t i : range(nstates)){
        hist.push_back(arma::zeros<vec>(bins[i].size());
      }
      for (vec sample : samples){
        for( size_t bin_i : range(nbins-1)){
          if(bins())

        }


      }

    }

    private:
    size_t nbins;
    vector<vec> cumul_distr;
    vector<vec>bins;
    vector<vec>bin_values;

    }
  
    
  }

}
