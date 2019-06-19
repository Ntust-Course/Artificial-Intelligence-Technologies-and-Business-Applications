"""Improvement to Bayesian"""
import matplotlib.pyplot as plt

from network import choice

if __name__ == "__main__":
    # Exploration method. Choose between: greedy, random, e-greedy, boltzmann, bayesian.

    # Some statistics on network performance

    bayesian_r_means, bayesian_j_means = choice("bayesian")
    # bayesian1_r_means, bayesian1_j_means = choice("bayesian-improvised")

    # plt.plot(r_means)

    plt.plot(bayesian_r_means)
    # plt.plot(bayesian1_r_means)
    plt.legend(["Bayesian", "Bayesian-Improved drop Out"], loc="upper left")
    plt.title("Comparison of different Exploration Approaches")
    plt.xlabel("Mean Rewards")
    plt.ylabel("Episodes")
