#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "Neuron.h"

class NeuralLayer {
 public:
  std::vector<Neuron> neurons_;

  NeuralLayer(int num_neurons = 0, int inputs_per_neuron = 0);
  ~NeuralLayer() = default;
  std::vector<double> forward(const std::vector<double>& inputs);
};