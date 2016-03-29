#pragma once

#include <tuple>
#include <tuple>
#include <boost/range/irange.hpp>
#include "armadillo"

using namespace std;
using namespace arma;

namespace mc{

  namespace utils{

    mat uniform(const double a, const double b, const size_t n_rows, size_t n_cols){
      // Matrix of x ~ U(a,b) size (n_rows,n_cols)
      return a + (b-a)*randu(n_rows,n_cols);

    }
    vec uniform(const double a, const double b, const size_t n){
      // Vector of x ~ U(a,b) size n
      return a + (b-a)*randu(n);

    }

    mat uniform(const double a, const double b, const SizeMat sizemat){
      // Matrix of x ~ U(a,b) size (n_rows,n_cols)
      return a + (b-a)*randu(sizemat);

    }

    mat uniform(const size_t & n_rows, const size_t & n_cols){
      // Matrix of x ~ U(0,1) size (n_rows,n_cols)
      return randu(n_rows,n_cols);

    }

    vec uniform(const size_t & n){
      return randu(n);
    }

    double uniform(){
      vec u = randu(1);
      return u(0);
    }


    mat norm(const double mu, const double sigma, const int n_rows, int n_cols){
      // Matrix of x ~ N(mu,sigma) size (n_rows,n_cols)
      return mu + sigma*randn(n_rows,n_cols);

    }

    mat norm(const size_t & n_rows, const size_t & n_cols){
      // Matrix of x ~ N(0,1) size (n_rows,n_cols)
      return randn(n_rows,n_cols);

    }

    vec norm(const size_t &n){
      // Matrix of x ~ N(0,1) size (n_rows,n_cols)
      return randn(n);

    }

    double norm(){
      // Matrix of x ~ N(0,1) size (n_rows,n_cols)
      vec n = randn(1); 
      return n(0);
    }

    int randint(int a, int b){
      // Random integer in [a,b]
      return randi<vec>(1,distr_param(a,b))(0);
    }

    size_t randint(const size_t & n){
      // Random integer in [0,n-1]
      return randi<vec>(1,distr_param(0,static_cast<int>(n-1)))(0);
    }


    template <typename T>
    auto range(T upper) -> decltype(boost::irange(static_cast<T>(0), upper)) {
      return boost::irange(static_cast<T>(0), upper);
    }

    template <typename T1, typename T2>
    auto range(T1 lower, T2 upper) -> decltype(boost::irange(static_cast<T2>(lower), upper)) {
      return boost::irange(static_cast<T2>(lower), upper);
    }

    /*

      Calculates all combinations of integer ranges of which the length is given in the dim vec.

     */
    Mat<size_t> combinations(const uvec & dim){
      size_t nvariables = dim.size();
      size_t ncombinations = prod(dim);
      Mat<size_t> result(ncombinations, nvariables);

      for(auto var_i : range(nvariables)){
        size_t var_val = 0;
        size_t var_len;

        if (var_i == nvariables-1){
          var_len = 1; // Last variable
        }else{
          var_len = prod(dim(span(var_i+1,nvariables-1)));
        }

        for(auto j : range(ncombinations)){
          if ((j % (var_len * dim(var_i)) == 0) && j > 0){
            var_val = 0;
          }else if ((j % var_len == 0) && j > 0){
            var_val += 1;
          }
          result(j, var_i) = var_val;
        }
      }
      return result;
    };

    /*! Use inverse uniform cdf sampling to sample from discrete distribution
     *
     *  \param Vector of probabilities that sum to one (density).
     *
     *  \return Index of the randomly sampled vector item.
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


    /*! Returns the possible actions for a given state.
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

    /*! Returns the action that maximizes the Q-value for the given state
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

    struct SoftStatePolicy{
      vec density; // Probability density for each possible action from this state
      uvec actions; // Possible actions from this state

      SoftStatePolicy(){};
      SoftStatePolicy(vec density, uvec actions) : density(density), actions(actions){};
      size_t sample(){
        return(actions(sample_discrete(this->density)));
      }
    };

    template <typename DiscretizedModelT>
    struct SoftPolicy{
    public:
      SoftPolicy(const DiscretizedModelT & discrete_model){
        size_t nstates = discrete_model.state_space_size;

        vector<SoftStatePolicy> soft_pol(nstates);

        for(auto state : range(nstates)){
          // Values for this state
          vec state_val = discrete_model.state_values.row(state);
          // Possible actions from this state
          uvec poss_a = possible_actions(state_val, discrete_model.actions, discrete_model.model);
          // Density with equal probabilities
          vec density = zeros(poss_a.size()) + 1.0/static_cast<double>(poss_a.size());
          // Instantiate policy
          SoftStatePolicy pol(density, poss_a);
          // Add to vector
          soft_pol[state] = pol;
        }
        this->policies = soft_pol;
      }

      SoftStatePolicy& operator[](const size_t & state){
        return this->policies[state];
      }

      SoftStatePolicy operator[](const size_t & state) const{
        return this->policies[state];
      }

      vector<SoftStatePolicy> policies;
    };
  }
}
