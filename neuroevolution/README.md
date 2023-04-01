# Neuroevolution with Evolutionary Computing in Python 🧠🔧💡

This repository contains a basic example of implementing a neuroevolution algorithm using evolutionary computing in Python 🐍. The goal is to train an artificial neural network (ANN) to solve the XOR problem by optimizing its weights and biases.

## XOR Problem 🧩

The XOR problem is a classic binary classification problem often used to evaluate the performance of neural networks. The XOR truth table is as follows:

| Input 1 | Input 2 | XOR Output |
|:------:|:------:|:----------:|
|   0    |   0    |     0      |
|   0    |   1    |     1      |
|   1    |   0    |     1      |
|   1    |   1    |     0      |

## How it works 🚀

The neuroevolution algorithm uses the DEAP library to implement an evolutionary algorithm that optimizes the weights and biases of a neural network with one hidden layer. The network has 2 input neurons, 5 hidden neurons, and 1 output neuron. The sigmoid activation function is used in the neurons.

The evolutionary algorithm includes the following genetic operations:

1. 🧬 Crossover: `cxBlend` with a blend factor `alpha` of 0.5.
2. 🦠 Mutation: `mutGaussian` with mean `mu` of 0, standard deviation `sigma` of 1, and mutation probability index `indpb` of 0.1.
3. 🎯 Selection: `selBest`, which selects the best individuals from the population.

The evolutionary algorithm runs for 100 generations with a population of 50 individuals.

## Results 📊

At the end of the run, the best individual found is displayed, containing the optimized weights and biases to solve the XOR problem. The statistical results of each generation are also shown, including mean, standard deviation, minimum, and maximum.

