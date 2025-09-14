#include "Neuron.h"

Neuron::Neuron(int inputs) {
    for (int i = 0; i < inputs; i++) {
        weights_.push_back((rand() % 100) / 100.0);
    }
}

double Neuron::activate(const std::vector<double>& inputs) {
    double sum{0};
    for (size_t i = 0; i < weights_.size(); i++) {
        sum += weights_[i] * inputs[i];
    }

    output_ = sigmoid(sum);
    return output_;
}