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


  }
}
