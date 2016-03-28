#pragma once

#include <stdexcept>
#include <vector>
#include <tuple>
#include <armadillo>
#include "mc-control/utils.hpp"
#include "mc-control/distribution.hpp"
#include "mc-control/problem.hpp"

using namespace std;
using namespace arma;
using namespace mc::utils;
using namespace mc::distributions;

namespace mc{

  namespace plot{

    template<typename ProblemT>
    void plot_q(const mat & Q, const ProblemT & problem){

      cout << "Plotting the Q-values!" << endl;

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
      // file << "fig = plt.figure()" << endl;
      // file << "ax = fig.add_subplot(111,projection='3d')" << endl;

      // // Colors
      // file << "colors = [hex['color'] for hex in list(plt.rcParams['axes.prop_cycle'])]" << endl;
      // file << "ncolors = len(colors)" << endl;
      // file << "coli = 0" << endl;

      // Write state and action values to arrays
      file << "state_values = []" << endl;
      for(auto state : range(Q.n_rows)){
        file << "state_values.append(" << problem.state_values(state) << ")" << endl;
      }

      file << "action_values = []" << endl;
      for(auto action  : range(Q.n_cols)){
        file << "action_values.append(" << problem.actions(action) << ")" << endl;
      }

      // Create Q-value matrix
      file << "Q = np.zeros((len(state_values), len(action_values)))" << endl;
      for(auto state  : range(Q.n_rows)){
        for(auto action  : range(Q.n_cols)){
          file << "Q[" << state << "," << action << "] = " << Q(state,action)<< endl;
        }
      }

      file << "X,Y = np.meshgrid(state_values, action_values)" << endl;
      file << "plt.pcolormesh(X,Y,Q.T)" << endl;

      file << "plt.xlabel('State')" << endl;
      file << "plt.ylabel('Action')" << endl;
      file << "plt.title('Q-values')" << endl;

      file << "plt.show()" << endl;

      file.close();

      system ("ipython plot.py");


    };

    void plot_distr(const vector<DiscreteDistribution> & distr, const vec & actions){

      cout << "Plotting the distribution!" << endl;
      // Only works for one-dim states for now
      if (distr[0].nvariables > 1){
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
