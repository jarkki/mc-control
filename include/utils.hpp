#pragma once

#include <tuple>
#include <boost/range/irange.hpp>
#include "armadillo"
//#include "/usr/local/Cellar/armadillo/6.400.3_1/include/armadillo"


using namespace std;
using namespace arma;

namespace mc{

  namespace utils{


    // Matrix of x ~ U(a,b) size (n_rows,n_cols)
    mat uniform(const double a, const double b, const int n_rows, int n_cols);
    vec uniform(const double a, const double b, const int n);
    vec uniform(const int n);
    // Matrix of x ~ U(a,b) size SizeMat
    mat uniform(const double a, const double b, const SizeMat sizemat);
    // Matrix of x ~ U(0,1) size (n_rows,n_cols)
    mat uniform(const int n_rows, int n_cols);
    // Matrix of x ~ N(mu,sigma) size (n_rows,n_cols)
    mat norm(const double mu, const double sigma, const int n_rows, int n_cols);
    // Matrix of x ~ N(0,1) size (n_rows,n_cols)
    mat norm(const int n_rows, int n_cols);
    vec norm(const int n);

    int randint(int a, int b);
    int randint(int n);
    double uniform();

    void plot(const mat & X, const vec & y);
    void plot_approx(const mat & X, const vec & y, const vec & yhat);
    void plot3d(const mat & X, const vec & y);

    tuple <mat,vec> create_test_data(double a, double b, double sd, int n);
    tuple <mat,vec> create_test_data_3d(double a, double b, double sd, int n);

    template<typename ApproxT>
    int argmax_q(const ApproxT & approximator,
                 const vec & x,
                 bool (*constraint)(const vec & x, const int & u )){


      // Use constraint to get possible actions at xnext and
      // then calculate Q at xnext for all possible actions
      vector<int> us;
      vector<double> q;
      size_t nactions = approximator.nactions;

      for (int u=0; u < nactions; ++u) {
        if (constraint(x,u)){
          us.push_back(u);
          q.push_back(approximator.qval(x,u));
        }}

      // Find the maxQ
      auto maxq = std::max_element(std::begin(q), std::end(q));
      auto maxqi = std::distance(std::begin(q), maxq);
      int maxu = us[maxqi];

      return maxu;

    };


    template<class Integer>
    boost::iterator_range< boost::range_detail::integer_iterator<Integer> > range(Integer n)
    {
      return boost::irange(0, n);
    }
    template<class Integer>
    boost::iterator_range< boost::range_detail::integer_iterator<Integer> > range(Integer start, Integer end)
    {
      return boost::irange(start, end);
    }

    Mat<int> combinations(const Col<int> & dim);


  }
}
