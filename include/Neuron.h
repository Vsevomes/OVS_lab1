#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

inline double dsigmoid(double y) {
    return y * (1.0 - y);
}

class Neuron {
public:
    double output_;
    double delta_;
    std::vector<double> weights_;

    Neuron(int inputs = 0);
    ~Neuron() = default;
    double activate(const std::vector<double>& inputs);
};