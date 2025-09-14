#include "NeuralLayer.h"

NeuralLayer::NeuralLayer(int num_neurons, int inputs_per_neuron) {
    for (int i = 0; i < num_neurons; i++) {
        neurons_.emplace_back(inputs_per_neuron);
    }
}

std::vector<double> NeuralLayer::forward(const std::vector<double>& inputs) {
        std::vector<double> outputs;
        for (auto& neuron : neurons_) {
            outputs.push_back(neuron.activate(inputs));
        }
        return outputs;
    }