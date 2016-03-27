#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include "utils.hpp"

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

    vec uniform(const int n){
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

    vec norm(const int n){
      // Matrix of x ~ N(0,1) size (n_rows,n_cols)
      return randn(n);

    }

    int randint(int a, int b){
      // Random integer in [a,b]
      return randi<vec>(1,distr_param(a,b))(0);
    }

    int randint(int n){
      // Random integer in [0,n-1]
      return randi<vec>(1,distr_param(0,n-1))(0);
    }


    double uniform(){
      vec u = randu(1);
      return u(0);
    }

    void plot(const mat & X, const vec & y){
      ofstream file;
      file.open ("plot.py");
      file << "import matplotlib.pyplot as plt" << endl;
      file << "x = []" << endl;
      file << "y = []" << endl;
      for (int i = 0; i < X.n_rows; i++) {
        file << "x.append(" << X(i,0) << ")" << endl;
        file << "y.append(" << y(i) << ")" << endl;
      }

      file << "plt.plot(x,y,'o')" << endl;
      file << "plt.show()" << endl;
      file.close();

      system ("python plot.py");
    }

    void plot_approx(const mat & X, const vec & y, const vec & yhat){
      ofstream file;
      file.open ("plot.py");
      file << "import matplotlib.pyplot as plt" << endl;
      file << "x = []" << endl;
      file << "y = []" << endl;
      file << "yhat = []" << endl;
      for (int i = 0; i < X.n_rows; i++) {
        file << "x.append(" << X(i,0) << ")" << endl;
        file << "y.append(" << y(i) << ")" << endl;
        file << "yhat.append(" << yhat(i) << ")" << endl;
      }

      file << "plt.plot(x,y,'o')" << endl;
      file << "plt.plot(x,yhat)" << endl;
      file << "plt.show()" << endl;
      file.close();

      system ("python plot.py");
    }


    void plot3d(const mat & X, const vec & y){
      ofstream file;
      file.open ("plot.py");
      file << "import matplotlib.pyplot as plt" << endl;
      file << "from plotting import *" << endl;
      file << "x = []" << endl;
      file << "y = []" << endl;
      file << "z = []" << endl;
      for (int i = 0; i < X.n_rows; i++) {
        file << "x.append(" << X(i,0) << ")" << endl;
        file << "y.append(" << X(i,1) << ")" << endl;
        file << "z.append(" << y(i) << ")" << endl;
      }

      file << "plot3d(x,y,z,type='scatter',show=True)" << endl;
      file << "plt.show()" << endl;
      file.close();

      system ("export PYTHONPATH=$PYTHONPATH:~/Dropbox/Koodia/github/fitq/fitq");
      system ("python plot.py");
    }


    tuple <mat,vec> create_test_data(double a, double b, double sd, int n){

      // Uniformly sampled points
      mat X = uniform(a,b,n,1);
      // Sort
      X = sort(X);
      // Sin with Gaussian noise
      vec y = arma::sin(X) + (norm(n,1) * sd);

      return make_tuple (X,y);
    }


    tuple <mat,vec> create_test_data_3d(double a, double b, double sd, int n){

      // Uniformly sampled points
      mat X = uniform(a,b,n,2);
      // Sin
      vec y = arma::sin(X.col(0)) % arma::sin(X.col(1)); // % is element-wise multiplication
      // Add noise
      y = y + (norm(n,1) * sd);
      return make_tuple (X,y);
    }

    Mat<size_t> combinations(const uvec & dim){
      size_t nvariables = dim.size();
      size_t ncombinations = prod(dim);
      Mat<size_t> result(ncombinations, nvariables);

      for(auto var_i : range(nvariables)){
        int var_val = 0;
        int var_len;

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
