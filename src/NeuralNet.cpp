#include "NeuralNet.h"

NeuralNet::NeuralNet(const std::string& filename, double learn_rate) {
  lr_ = learn_rate;

  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Model file open error!" << std::endl;
    return;
  }

  int param;
  for (size_t i = 0; i < 4; i++) {
    file >> param;
    topology_.push_back(param);
  }

  // create network
  layers_.clear();
  layers_.emplace_back(topology_[2], topology_[0]);

  for (size_t i = 1; i < topology_[1]; i++) {
    layers_.emplace_back(topology_[2], topology_[2]);
  }

  layers_.emplace_back(topology_[3], topology_[2]);

  // load weights
  for (auto& layer : layers_) {
    for (auto& neuron : layer.neurons_) {
      for (auto& w : neuron.weights_) file >> w;
    }
  }
  file.close();
}

NeuralNet::NeuralNet(int input_size, int hidden_layers, int hidden_neurons,
                     int output_size, double learn_rate) {
  lr_ = learn_rate;

  topology_.push_back(input_size);
  topology_.push_back(hidden_layers);
  topology_.push_back(hidden_neurons);
  topology_.push_back(output_size);

  layers_.emplace_back(topology_[2], topology_[0]);

  for (size_t i = 1; i < topology_[1]; i++) {
    layers_.emplace_back(topology_[2], topology_[2]);
  }

  layers_.emplace_back(topology_[3], topology_[2]);
}

std::vector<double> NeuralNet::forward(const std::vector<double>& input) {
  std::vector<double> activations = input;
  for (auto& layer : layers_) {
    activations = layer.forward(activations);
  }
  return activations;
}

void NeuralNet::train(const std::vector<double>& input,
                      const std::vector<double>& target) {
  std::vector<double> activations = forward(input);

  // exit layer error
  NeuralLayer& output_layer = layers_.back();
  for (size_t i = 0; i < output_layer.neurons_.size(); i++) {
    double out = output_layer.neurons_[i].output_;
    double error = target[i] - out;
    output_layer.neurons_[i].delta_ = error * dsigmoid(out);
  }

  // hidden layers errors
  for (int l = layers_.size() - 2; l >= 0; l--) {
    NeuralLayer& current = layers_[l];
    NeuralLayer& next = layers_[l + 1];
    for (size_t i = 0; i < current.neurons_.size(); i++) {
      double error = 0.0;
      for (auto& n : next.neurons_) {
        error += n.weights_[i] * n.delta_;
      }
      current.neurons_[i].delta_ =
          error * dsigmoid(current.neurons_[i].output_);
    }
  }

  // weight update
  std::vector<double> prev_activations = input;
  for (size_t l = 0; l < layers_.size(); l++) {
    NeuralLayer& layer = layers_[l];
    if (l > 0) {
      prev_activations.clear();
      for (auto& n : layers_[l - 1].neurons_) {
        prev_activations.push_back(n.output_);
      }
    }

    for (auto& neuron : layer.neurons_) {
      for (size_t w = 0; w < neuron.weights_.size(); w++) {
        neuron.weights_[w] += lr_ * neuron.delta_ * prev_activations[w];
      }
    }
  }
}

void NeuralNet::save(const std::string& filename) {
  std::ofstream file(filename);
  if (!file) {
    std::cerr << "Save file open error!" << std::endl;
    return;
  }

  // // сохраняем топологию
  // file << topology.size() << "\n";
  for (int t : topology_) file << t << " ";
  file << "\n";

  // сохраняем веса
  for (auto& layer : layers_) {
    for (auto& neuron : layer.neurons_) {
      for (auto w : neuron.weights_) file << w << " ";
    }
  }
  file.close();
}