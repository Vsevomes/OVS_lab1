#pragma once

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "NeuralLayer.h"

#define LR 0.1

class NeuralNet {
 public:
  std::vector<NeuralLayer> layers_;
  std::vector<int> topology_;

  NeuralNet(const std::string& filename, double learn_rate = LR);
  NeuralNet(int input_size, int hidden_layers, int hidden_neurons,
            int output_size, double learn_rate = LR);
  ~NeuralNet() = default;
  std::vector<double> forward(const std::vector<double>& input);
  void train(const std::vector<double>& input,
             const std::vector<double>& target);
  void save(const std::string& filename);

 private:
  double lr_;
};