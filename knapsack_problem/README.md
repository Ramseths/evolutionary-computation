# Knapsack Problem Genetic Algorithm in Python 🎒🔧💡

This repository contains a basic implementation of a genetic algorithm to solve the Knapsack Problem in Python 🐍. The goal is to select items with different weights and values to maximize the total value without exceeding a given weight limit.

## Knapsack Problem 🎒

The Knapsack Problem is a combinatorial optimization problem, where the objective is to select items with different weights and values in such a way that the total value of the selected items is maximized, while the total weight does not exceed a predefined weight limit.

## How it works 🚀

The genetic algorithm uses the DEAP library to implement an evolutionary algorithm that optimizes the selection of items for the knapsack problem. The algorithm consists of the following genetic operations:

1. 🧬 Crossover: `cxTwoPoint`
2. 🦠 Mutation: `mutFlipBit` with mutation probability index `indpb` of 0.1.
3. 🎯 Selection: `selBest`

The evolutionary algorithm runs for 100 generations with a population of 50 individuals.

## Results 📊
At the end of the run, the best individual found is displayed, containing the optimized selection of items for the Knapsack Problem.