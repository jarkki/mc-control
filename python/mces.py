import pdb
import numpy as np
from numpy.random import randint, uniform
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
from itertools import cycle
from scipy.interpolate import griddata, RectBivariateSpline, SmoothBivariateSpline, LinearNDInterpolator
from scipy.optimize import fminbound

# Model parameters
theta = 0.5
alpha = 0.8
rho = 0.9
df = rho

# State space bins
# nxbins = 100
xlim = [0.0, 8.0]
# xbins = np.linspace(xlim[0], xlim[1], nxbins+1)
# Calulate middle values for each bin
# states = [(xbins[i] + xbins[i+1])/2.0 for i in range(len(xbins)-1)]

# Actions
nactions = 30
actions = np.linspace(xlim[0], xlim[1],nactions)

# gamma = rho
# nepisodes = 50000
# niterations = 10
# asamples = 1000

# System dynamics
def U(c): return 1 - np.exp(- theta * c)
def reward(x,u,xnext): return 1 - np.exp(- theta * (x-u))  # Utility
def trans(x,u,z): return (u**alpha) * z     # Transition function for the income process

# Constraint 0.0 <= u <= x
def constraint(u,x):
    return xlim[0] <= u <= x

# Test for actions[1]
# Discretize the distribution
class InvCdfSampler:
    def __init__(self, samples, states, bins):

        nbins = len(states)

        # Create histogram
        hist = np.zeros(nbins)
        for sample in samples:
            for i in range(nbins-1):
                if bins[i] <= sample < bins[i+1]:
                    hist[i] += 1
                    break

        # Normalize histogram so that the integral over the range is 1
        density = np.zeros(nbins)
        dx = bins[1] - bins[0]
        density = hist/(sum(hist) * dx)

        # Calc cumulative distribution function
        cum = np.zeros(len(bins))
        cum[1:] = np.cumsum(density*dx)

        self.cum = cum
        self.bins = bins
        self.bin_centers = states
        self.nbins = nbins

    def sample(self,n=1,return_indices=True):
        samples = []
        us = uniform(size=n)
        for u in us:
            for i in range(self.nbins):
                if self.cum[i] <= u < self.cum[i+1]:
                    samples.append(i)
                    break

        if n == 1:
            if return_indices:
                return samples[0]
            else:
                return self.bin_centers[samples[0]]
        else:
            if return_indices:
                return samples
            else:
                return [self.bin_centers[s] for s in samples]

    def hist(self, samples):
        # Create histogram
        hist = np.zeros(self.nbins)
        for sample in samples:
            for i in range(self.nbins-1):
                if self.bins[i] <= sample < self.bins[i+1]:
                    hist[i] += 1
                    break

        # Normalize histogram so that the integral over the range is 1
        density = np.zeros(self.nbins)
        dx = self.bins[1] - self.bins[0]
        density = hist/(sum(hist) * dx)

        return (self.bins[0:nbins], density, dx)

def plot_state_action_densities(densities,actions):
    # Initiate plotting
    plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    # Colors
    colors = [hex['color'] for hex in list(plt.rcParams['axes.prop_cycle'])]
    ncolors = len(colors)
    coli = 0

    # Plot
    for action in range(len(densities)):
        # Sample from the sampler
        samples = densities[action].sample(10000,return_indices=False)
        # Get the histogram
        bins, density, width = densities[action].hist(samples)
        # Plot the sample
        ax.bar(bins, density, np.zeros(len(bins))+actions[action], zdir='y',  alpha=0.8, color=colors[coli], width=width)
        # Cycle colors
        if coli >= (ncolors-1):
            coli = 0
        else:
            coli += 1

    ax.set_xlabel('State')
    ax.set_ylabel('Action')
    ax.set_zlabel('p(state)')
    plt.show()

# Create bins for state space
nbins = 30
state_bins = np.linspace(xlim[0], xlim[1], nbins+1)
# Calculate middle values for each bin
state_values = [(state_bins[i] + state_bins[i+1])/2.0 for i in range(len(state_bins)-1)]
# Realizations of log-norm distr.
W = np.exp(norm.rvs(size=100000))

# Create state-action densities
densities = []
# For each action
for action in range(nactions):
    # Sample the transition function
    trans_sample = trans(0.0, actions[action], W)
    # Create inverse cdf sampler instance from empirical transition sample
    sampler = InvCdfSampler(trans_sample, state_values, state_bins)
    densities.append(sampler)

# Plot the densities
# plot_state_action_densities(densities,actions)

def argmax_q(actions,qvals):
    if len(actions) != len(qvals):
        throw ("Length of actions and qvals -arrays must match!")
    # Calculate the maximum Q-value
    maxq = np.max(qvals)
    # Check if there are several actions with same q-value as maxq
    maxq_actions = [a for a,q in zip(actions,qvals) if q == maxq]
    if len(maxq_actions) > 1:
        # Return randomly one of the actions that maximize the Q-value
        return np.random.choice(maxq_actions)
    else:
        # Return the single action that maximizes the Q-value
        return maxq_actions[0]

def mc_es(densities, state_bins, state_values, actions, niterations=10000, epsilon=0.1):
    """ Monte Carlo reinforcement learning with Exloring Starts (MC-ES)
    """
    nstates = len(state_values)
    nactions = len(actions)
    def possible_actions(state):
        return [a for  a in range(nactions) if constraint(actions[a], state_values[state])]

    # Q-action-value table
    Q = np.zeros((nstates, nactions))
    # Counter for explored state-actions
    counter = np.zeros((nstates,nactions))
    # Rewards
    rewards = np.zeros((nstates,nactions))
    # Start iterating
    for iteration in range(niterations):
        # Draw random starting state
        state = randint(nstates)
        # Choose action (epsilon-greedy)
        poss_actions = possible_actions(state)
        if uniform() < epsilon:
            action = np.random.choice(poss_actions)
        else:
            qvals = [Q[state, a] for a in poss_actions]
            action = argmax_q(poss_actions, qvals)
        # Transition to next state
        next_state = densities[action].sample()
        # Increase count for this state-action pair
        counter[state,action] += 1
        # Calculate reward and Increase the sum of rewards for this state-action pair
        rewards[state,action] += U(state_values[state] - actions[action]) + df * U(densities[action].sample(return_indices=False))
        # Update the Q-value for this state-action pair
        Q[state,action] = rewards[state,action]/counter[state,action]
        # Print info
        if iteration % 1000 == 0:
            print ("Iter. {}".format(iteration))
    return Q

# Run it
Q = mc_es(densities, state_bins, state_values, actions, niterations=5000000, epsilon=0.1)

# Plot
X,Y = np.meshgrid(state_values, actions)
plt.pcolormesh(X,Y,Q.T)
states = range(nbins)
pol = []
for state in states:
    pol.append(actions[np.argmax(Q[state,:])])
adx = actions[1]-actions[0]
sdx = state_values[1] - state_values[0]
plt.plot([state_values[state] for state in states] + sdx/2, pol + adx/2, linewidth=3.0, color='white')
x = np.linspace(xlim[0],xlim[1],100)
plt.plot(x,x)
plt.show()


