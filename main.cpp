#include <iostream>
#include <filesystem>
#include <sstream>
#include <algorithm>
#include <chrono>
#include "NeuralNet.h"

#define TARGET_ERROR 0.01
#define MAX_EPOCHES 1000000

#include <chrono>
#include <iostream>

using Clock = std::chrono::high_resolution_clock;
using ms = std::chrono::milliseconds;
using mc = std::chrono::microseconds;

struct Sample {
    std::vector<double> input;
    std::vector<double> target;
};

std::vector<Sample> loadSamplesFromFolder(const std::string& folder);
std::vector<double> loadSampleFromFile(const std::string& filename);
void menu(std::string prog_name);

int main(int argc, char* argv[]) {
    srand(time(NULL));

    if (argc < 2) {
        menu(argv[0]);
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "train") {
        if (argc < 7) {
            std::cerr << "Need more arguments" << std::endl;
            return 1;
        }

        std::string train_folder = argv[2];
        int num_inputs = std::stoi(argv[3]);
        int hidden_layers = std::stoi(argv[4]);
        int neurons_per_layer = std::stoi(argv[5]);
        int num_outputs = std::stoi(argv[6]);

        NeuralNet net(num_inputs, hidden_layers, neurons_per_layer, num_outputs);
        
        std::vector<Sample> samples = loadSamplesFromFolder(train_folder);
        if (samples.empty()) {
            std::cerr << "No test data" << std::endl;
            return 1;
        }

        // training settings
        double err = 1.0;         
        int epoch = 0;

        auto start_train = Clock::now();

        // training process
        while (err > TARGET_ERROR && epoch < MAX_EPOCHES) {
            err = 0.0;
            for (auto &s : samples) {
                net.train(s.input, s.target);
                std::vector<double> out = net.forward(s.input);
                for (int i = 0; i < num_outputs; i++)
                    err += (s.target[i] - out[i]) * (s.target[i] - out[i]);
            }
            err /= 2;
            epoch++;
            if (epoch % 1000 == 0) {
                std::cout << "Epoch " << epoch << ", ERR = " << err << std::endl;
            }
        }

        // time and memory count section
        auto end_train = Clock::now();
        auto duration_train = std::chrono::duration_cast<ms>(end_train - start_train).count();

        auto start_forward = Clock::now();
        auto out = net.forward(samples[0].input);
        auto end_forward = Clock::now();
        auto duration_forward = std::chrono::duration_cast<mc>(end_forward - start_forward).count();

        size_t memory_usage = 0;
        for (auto &layer : net.layers_) {
            memory_usage += layer.neurons_.size() * sizeof(Neuron);
            for (auto &neuron : layer.neurons_)
                memory_usage += neuron.weights_.size() * sizeof(double);
        }
    
        std::cout << "Learning completed for " << epoch << " epochs, ERR = " << err << std::endl;
        std::cout << "Learning time: " << duration_train << " ms" << std::endl;
        std::cout << "Network exit time: " << duration_forward << " mc" << std::endl;
        std::cout << "Memory usage: " << memory_usage << " bites" << std::endl;

        std::string save_file = "../data/model";

        net.save(save_file);
    } else if (mode == "predict") {
        if (argc < 4) {
            std::cerr << "Need more arguments" << std::endl;
            return 1;
        }

        std::string model_file = argv[2];
        std::string input_file = argv[3];

        NeuralNet net(model_file);

        std::vector<double> input = loadSampleFromFile(input_file);
        if (input.empty()) {
            std::cerr << "Error: empty data" << std::endl;
            return 1;
        }

        // start
        std::vector<double> out = net.forward(input);

        int predicted = std::max_element(out.begin(), out.end()) - out.begin();

        if (out[0] == out[predicted] && out[predicted] > 0.8) {
            std::cout << "Shape: circle" << std::endl << "Accuracy: " << out[predicted] << std::endl;
        }
        else if (out[1] == out[predicted] && out[predicted] > 0.8) {
            std::cout << "Shape: rectangle" << std::endl << "Accuracy: " << out[predicted] << std::endl;
        }
        else if (out[2] == out[predicted] && out[predicted] > 0.8) {
            std::cout << "Shape: triangle" << std::endl << "Accuracy: " << out[predicted] << std::endl;
        }
        else {
            std::cout << "Unkown shape" << std::endl;
        }
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }
    return 0;
}

std::vector<Sample> loadSamplesFromFolder(const std::string& folder) {
    std::vector<Sample> samples;

    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().string();

        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Open file error" << filename << std::endl;
            continue;
        }

        Sample sample;

        std::string line;
        if (getline(file, line)) {
            sample.target.clear();
            for (char c : line) {
                if (c == '0' || c == '1') sample.target.push_back(c - '0');
            }
        } else {
            std::cerr << "Empty file" << filename << std::endl;
            continue;
        }

        sample.input.clear();
        while (getline(file, line)) {
            if (line.empty()) continue;
            for (char c : line) {
                if (c == '0' || c == '1') sample.input.push_back(c - '0');
            }
        }

        file.close();
        samples.push_back(sample);
    }

    return samples;
}

std::vector<double> loadSampleFromFile(const std::string& filename) {
    std::vector<double> input;
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Ошибка открытия файла " << filename << std::endl;
        return input;
    }

    std::string line;
    // skip first target
    getline(file, line);

    while (getline(file, line)) {
        if (line.empty()) continue;
        for (char c : line) {
            if (c == '0' || c == '1') input.push_back(c - '0');
        }
    }

    file.close();
    return input;
}

void menu(std::string prog_name){
    std::cerr << "Usage:" << std::endl;
        std::cerr << "  For train model: " 
             << prog_name << " train <train_folder> <inputs> <hiddenLayers> <neuronsPerLayer> <outputs>" << std::endl;
        std::cerr << "  For use model: " 
             << prog_name << " predict <model_file> <input_file>" << std::endl;
}