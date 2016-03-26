#pragma once

#include <stdexcept>
#include <vector>
#include <tuple>
#include <armadillo>
#include "utils.hpp"

using namespace std;
using namespace arma;
using namespace mc::utils;

namespace mc{

  namespace distributions{

    /*
      Creates a discrete distribution from a given sample (continuous or discrete) and bins.
      Allows one to draw samples from the resulting discretized distribution.
    */
    class DiscreteDistribution{
    public:

      // TODO: Constructor that constructs the discrete distribution from continuous density function

      DiscreteDistribution(mat samples, vector<vec> bins, vector<vec> bin_values){

        // TODO: Check that the the bins are equally spaced

        int nstates = samples.n_cols;;
        int nsamples = samples.n_rows;

        // Initialize histograms to zero
        vector<vec> hists;
        for(int state : range(nstates)){
          vec hist = arma::zeros(bins[state].size()-1);
          hists.push_back(hist);
        }

        // Create histogram for each state variable
        for (auto sample : range(nsamples)){
          // For each state variable in this sample
          for(int state : range(nstates)){
            // For each bin
            int bin_size = bins[state].size()-1;
            for (int bin_i : range(bin_size)){
              auto state_value = samples(sample,state);
              // If state variable value falls into this bin, increase histogram value for this bin
              if((bins[state](bin_i) <= state_value) &&
                 (state_value < bins[state](bin_i+1))){
                hists[state](bin_i) += 1.0;
                break;
              }
            }
          }
        }

        // Create cumulative distributions and densities
        vector<vec> cumul_distrs;
        vector<vec> densities;
        vec bin_widths(nstates);
        for(int state : range(nstates)){

          // Normalize the histograms integrate to 1 (create densities)
          vec density = arma::zeros(bins[state].size()-1);
          double dx = bins[state](1) - bins[state](0);
          density = hists[state]/(arma::sum(hists[state]) * dx);

          // Calculate cumulative distribution function
          vec cum_distr = arma::zeros(bins[state].size());
          cum_distr(span(1,cum_distr.size()-1)) = arma::cumsum(density * dx);

          cumul_distrs.push_back(cum_distr);
          densities.push_back(density);
          bin_widths(state) = dx;
        }

        // Calculate the number of bins for each variable
        vector<int> nbins(bins.size());
        for(int state : range(nstates)){
          nbins[state] = bins[state].size();
        }

        this->nstates = bins.size();
        this->cumul_distrs = cumul_distrs;
        this->nbins = nbins;
        this->bins = bins;
        this->bin_values = bin_values;
        this->bin_widths = bin_widths;
        this->densities = densities;
      }

      Mat<int> sample_indices(int n=1){
        Mat<int> sample_indices(n,this->nstates);
        for(auto i : range(n)){
          auto u = uniform(this->nstates);
          for(auto state : range(this->nstates)){
            for(auto bin_i : range(this->nbins[state])){
              if((this->cumul_distrs[state](bin_i) <= u(state)) &&
                 (u(state) < this->cumul_distrs[state](bin_i+1))){
                sample_indices(i,state) = bin_i;
              }}}}
        return sample_indices;

      };

      mat sample_values(int n=1){
        mat sample_values(n,this->nstates);
        for(auto i : range(n)){
          auto u = uniform(this->nstates);
          for(auto state : range(this->nstates)){
            for(auto bin_i : range(this->nbins[state])){
              if((this->cumul_distrs[state](bin_i) <= u(state)) &&
                 (u(state) < this->cumul_distrs[state](bin_i+1))){
                sample_values(i,state) = this->bin_values[state](bin_i);
              }}}}

        return sample_values;
      };

      int nstates;
      vector<int> nbins;
      vector<vec> cumul_distrs;
      vector<vec> bins;
      vector<vec> bin_values;
      vec bin_widths;
      vector<vec> densities;
    };




    void plot_distr(const vector<DiscreteDistribution> & distr, const vec & actions){

      cout << "Plotting!" << endl;
      // Only works for one-dim states for now
      if (distr[0].nstates > 1){
        throw invalid_argument("Plotting only works on one-dimensional states for now...");
      }
      // Open file
      ofstream file;
      file.open ("plot.py");

      // Imports
      file << "import numpy as np" << endl;
      file << "import matplotlib.pyplot as plt" << endl;
      file << "from matplotlib import cm" << endl;
      file << "from mpl_toolkits.mplot3d import Axes3D" << endl;

      // Initiate plotting
      file << "plt.style.use('ggplot')" << endl;
      file << "fig = plt.figure()" << endl;
      file << "ax = fig.add_subplot(111,projection='3d')" << endl;

      // Colors
      file << "colors = [hex['color'] for hex in list(plt.rcParams['axes.prop_cycle'])]" << endl;
      file << "ncolors = len(colors)" << endl;
      file << "coli = 0" << endl;

      // For each action
      int nactions = actions.size();
      for (auto action : range(nactions)){
          file << "bins = []" << endl;
          file << "density = []" << endl;
          file << "width = " << distr[action].bin_widths(0) << endl;

          for(auto bin_i : range(distr[action].nbins[0]-1)){
            file << "bins.append(" << distr[action].bins[0](bin_i) << ")" << endl;
            file << "density.append(" << distr[action].densities[0](bin_i) << ")" << endl;
          }

          file << "ax.bar(bins, density, np.zeros(len(bins))+" << actions(action) << ", zdir='y',  alpha=0.8, color=colors[coli], width=width)" << endl;

          // Cycle colors
          file << "if coli >= (ncolors-1):" << endl;
          file << "    coli = 0" << endl;
          file << "else:" << endl;
          file << "    coli += 1" << endl;
        }
      file << "ax.set_xlabel('State')" << endl;
      file << "ax.set_ylabel('Action')" << endl;
      file << "ax.set_zlabel('Density')" << endl;
      file << "plt.show()" << endl;

      file.close();

      system ("ipython plot.py");


    }
  }

}
