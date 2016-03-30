#pragma once

#include <tuple>
#include <tuple>
#include <boost/range/irange.hpp>
#include "armadillo"

using namespace std;
using namespace arma;

namespace mc{

  namespace utils{

    //! Matrix of x ~ U(a,b) size (n_rows,n_cols)
    mat uniform(const double a, const double b, const size_t n_rows, size_t n_cols){
      return a + (b-a)*randu(n_rows,n_cols);
    }

    //! Vector of x ~ U(a,b) size n
    vec uniform(const double a, const double b, const size_t n){
      return a + (b-a)*randu(n);
    }

    //! Matrix of x ~ U(a,b) size (n_rows,n_cols)
    mat uniform(const double a, const double b, const SizeMat sizemat){
      return a + (b-a)*randu(sizemat);
    }

    //! Matrix of x ~ U(0,1) size (n_rows,n_cols)
    mat uniform(const size_t & n_rows, const size_t & n_cols){
      return randu(n_rows,n_cols);
    }
    //! vector of x ~ U(0,1) size n
    vec uniform(const size_t & n){
      return randu(n);
    }

    //! Single draw from U(0,1)
    double uniform(){
      vec u = randu(1);
      return u(0);
    }

    //! Matrix of x ~ N(mu,sigma) size (n_rows,n_cols)
    mat norm(const double mu, const double sigma, const int n_rows, int n_cols){
      return mu + sigma*randn(n_rows,n_cols);
    }

    //! Matrix of x ~ N(0,1) size (n_rows,n_cols)
    mat norm(const size_t & n_rows, const size_t & n_cols){
      return randn(n_rows,n_cols);
    }

    //! Vector of x ~ N(0,1) size n
    vec norm(const size_t &n){
      return randn(n);
    }

    //! Single draw from N(0,1)
    double norm(){

      vec n = randn(1);
      return n(0);
    }

    //! Random integer in [a,b]
    int randint(int a, int b){
      return randi<vec>(1,distr_param(a,b))(0);
    }

    // Random integer in [0,n-1]
    size_t randint(const size_t & n){
      return randi<vec>(1,distr_param(0,static_cast<int>(n-1)))(0);
    }

    /*! Integer range in [0,upper-1]
     *
     *
     *  Uses boost::irange to provide iterable range of integers.
     *
     *  @param upper : upper limit for the range
     *
     *  @retval boost::irange
     *
     *  Example usage:
     *  @code
     *   size_t n = 10;
     *   for(auto i : range(n)){
     *     cout << i << endl;
     *   }
     *  @endcode
     */
    template <typename T>
    auto range(T upper) -> decltype(boost::irange(static_cast<T>(0), upper)) {
      return boost::irange(static_cast<T>(0), upper);
    }

    /*! Integer range in [lower,upper-1]
     *
     *
     *  Uses boost::irange to provide iterable range of integers.
     *
     *  @param lower : lower limit for the range
     *  @param upper : upper limit for the range
     *
     *  @retval boost::irange
     *
     *  Example usage:
     *  @code
     *   size_t n = 10;
     *   for(auto i : range(1,n)){
     *     cout << i << endl;
     *   }
     *  @endcode
     */
    template <typename T1, typename T2>
    auto range(T1 lower, T2 upper) -> decltype(boost::irange(static_cast<T2>(lower), upper)) {
      return boost::irange(static_cast<T2>(lower), upper);
    }

    /*! Combinations of integer ranges
     *
     *  Calculates all combinations of integer ranges of which the length is given in the input vector.
     *
     *  @param dim   vector of dimensions for each variable
     *
     *  @retval      Matrix of size (prod(dim) x #variables)
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

    /*! Inverse uniform cdf sampling to sample from discrete distribution
     *
     *
     *  https://en.wikipedia.org/wiki/Inverse_transform_sampling
     *
     *  @param densities Vector of discrete probabilities that sum to one (density)
     *
     *  @retval Index of the randomly sampled vector item.
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
      // Shouldn't get to here
      throw invalid_argument("Function sample_discrete failed...");
    }


    /*! Returns a random policy, given the possible actions for each state.
     *
     */
    uvec create_random_policy(const vector<uvec> & possible_actions){
      size_t nstates = possible_actions.size();
      uvec pol(nstates);
      for(auto state : range(nstates)){
        pol(state) = possible_actions[state](randint(possible_actions[state].size()));
      }
      return pol;
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

    template<typename DiscretizedModelT>
    vector<uvec> create_possible_actions_matrix(const DiscretizedModelT & discrete_model){

      vector<uvec> possible;
      for(auto state : range(discrete_model.state_space_size)){
        vec state_value = discrete_model.state_values.row(state);
        vector<size_t> possible_for_state;
        for(auto action : range(discrete_model.actions.size())){
          if(discrete_model.model.constraint(discrete_model.actions(action), state_value)){
            possible_for_state.push_back(action);
          }
        }
        uvec state_actions = conv_to<uvec>::from(possible_for_state);
        possible.push_back(state_actions);
      }
      return possible;

    }

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

  }
}
