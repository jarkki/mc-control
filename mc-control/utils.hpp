#pragma once

#include <tuple>
#include <tuple>
#include <boost/range/irange.hpp>
#include "armadillo"

using namespace std;
using namespace arma;

namespace mc{

  namespace utils{

    mat uniform(const double a, const double b, const int n_rows, int n_cols){
      // Matrix of x ~ U(a,b) size (n_rows,n_cols)
      return a + (b-a)*randu(n_rows,n_cols);

    }
    vec uniform(const double a, const double b, const int n){
      // Vector of x ~ U(a,b) size n
      return a + (b-a)*randu(n);

    }

    vec uniform(const size_t n){
      return randu(n);
    }

    mat uniform(const double a, const double b, const SizeMat sizemat){
      // Matrix of x ~ U(a,b) size (n_rows,n_cols)
      return a + (b-a)*randu(sizemat);

    }

    mat uniform(const int n_rows, int n_cols){
      // Matrix of x ~ U(0,1) size (n_rows,n_cols)
      return randu(n_rows,n_cols);

    }

    mat norm(const double mu, const double sigma, const int n_rows, int n_cols){
      // Matrix of x ~ N(mu,sigma) size (n_rows,n_cols)
      return mu + sigma*randn(n_rows,n_cols);

    }

    mat norm(const int n_rows, int n_cols){
      // Matrix of x ~ N(0,1) size (n_rows,n_cols)
      return randn(n_rows,n_cols);

    }

    vec norm(const size_t n){
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

    int randint(size_t n){
      // Random integer in [0,n-1]
      return randi<vec>(1,distr_param(0,n-1))(0);
    }


    double uniform(){
      vec u = randu(1);
      return u(0);
    }

    // vector<size_t> range(const size_t & n){
    //   vector<size_t> v(n);
    //   for(size_t i = 0; i < n; ++i){
    //     v[i] = i;
    //   }
    //   return v;
    // }

    // template<class size_t>
    // boost::iterator_range< boost::range_detail::integer_iterator<size_t> > range(size_t  n)
    // {
    //   return boost::irange(0, n);
    // }

    template <typename T>
    auto range(T upper) -> decltype(boost::irange(static_cast<T>(0), upper)) {
      return boost::irange(static_cast<T>(0), upper);
    }

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


    // Matrix of x ~ U(a,b) size (n_rows,n_cols)
    // mat uniform(const double a, const double b, const int n_rows, int n_cols);
    // vec uniform(const double a, const double b, const int n);
    // vec uniform(const int n);
    // // Matrix of x ~ U(a,b) size SizeMat
    // mat uniform(const double a, const double b, const SizeMat sizemat);
    // // Matrix of x ~ U(0,1) size (n_rows,n_cols)
    // mat uniform(const int n_rows, int n_cols);
    // // Matrix of x ~ N(mu,sigma) size (n_rows,n_cols)
    // mat norm(const double mu, const double sigma, const int n_rows, int n_cols);
    // // Matrix of x ~ N(0,1) size (n_rows,n_cols)
    // mat norm(const int n_rows, int n_cols);
    // vec norm(const int n);

    // int randint(int a, int b);
    // int randint(int n);
    // double uniform();

    // void plot(const mat & X, const vec & y);
    // void plot_approx(const mat & X, const vec & y, const vec & yhat);
    // void plot3d(const mat & X, const vec & y);

    // tuple <mat,vec> create_test_data(double a, double b, double sd, int n);
    // tuple <mat,vec> create_test_data_3d(double a, double b, double sd, int n);

    // template<typename ApproxT>
    // int argmax_q(const ApproxT & approximator,
    //              const vec & x,
    //              bool (*constraint)(const vec & x, const int & u )){


    //   // Use constraint to get possible actions at xnext and
    //   // then calculate Q at xnext for all possible actions
    //   vector<int> us;
    //   vector<double> q;
    //   size_t nactions = approximator.nactions;

    //   for (int u=0; u < nactions; ++u) {
    //     if (constraint(x,u)){
    //       us.push_back(u);
    //       q.push_back(approximator.qval(x,u));
    //     }}

    //   // Find the maxQ
    //   auto maxq = std::max_element(std::begin(q), std::end(q));
    //   auto maxqi = std::distance(std::begin(q), maxq);
    //   int maxu = us[maxqi];

    //   return maxu;

    // };


    // Mat<size_t> combinations(const uvec & dim);



    // template<class size_t>
    // boost::iterator_range< boost::range_detail::integer_iterator<size_t> > range2(size_t  n)
    // {
    //   return boost::irange(0, n);
    // }

    // template<class Integer>
    // boost::iterator_range< boost::range_detail::integer_iterator<Integer> > range(Integer n)
    // {
    //   return boost::irange(0, n);
    // }
    // template<class Integer>
    // boost::iterator_range< boost::range_detail::integer_iterator<Integer> > range(Integer start, Integer end)
    // {
    //   return boost::irange(start, end);
    // }



  }
}
