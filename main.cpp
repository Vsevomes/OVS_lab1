#include <iostream>
#include <filesystem>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include "NeuralNet.h"

#define TARGET_ERROR 0.01
#define MAX_EPOCHES 100000

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
void drawFigure(std::vector<double>& figure);

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
    
        std::cout << "Training completed for " << epoch << " epochs, ERR = " << err << std::endl;
        std::cout << "Train time: " << duration_train << " ms" << std::endl;
        std::cout << "Inference time: " << duration_forward << " mcs" << std::endl;
        std::cout << "Memory usage: " << memory_usage << " bites" << std::endl;

        std::string save_file = "../data/model";

        net.save(save_file);
    } else if (mode == "predict") {
        if (argc < 4) {
            std::cerr << "Need more arguments" << std::endl;
            return 1;
        }

        std::string model_file = argv[2];
        std::string input_folder = argv[3];

        NeuralNet net(model_file);

        std::vector<Sample> inputs = loadSamplesFromFolder(input_folder);
        if (inputs.empty()) {
            std::cerr << "Error: empty data" << std::endl;
            return 1;
        }

        int num_outputs = inputs[0].target.size();
        std::vector<std::vector<int>> confusion(num_outputs, std::vector<int>(num_outputs, 0));

        // start
        for (auto& i: inputs) {
            std::vector<double> out = net.forward(i.input);
            drawFigure(i.input);

            int predicted = std::max_element(out.begin(), out.end()) - out.begin();
            int actual    = std::max_element(i.target.begin(), i.target.end()) - i.target.begin();

            confusion[actual][predicted]++;

            if (predicted == 0) {
                std::cout << "Shape: circle (p=" << out[predicted] << ")" << std::endl;
            }
            else if (predicted == 1) {
                std::cout << "Shape: rectangle (p=" << out[predicted] << ")" << std::endl;
            }
            else if (predicted == 2) {
                std::cout << "Shape: triangle (p=" << out[predicted] << ")" << std::endl;
            }
            else {
                std::cout << "Unknown shape" << std::endl;
            }

            std::cout << std::endl;
        }

        // metrics
        int total = 0, correct = 0;
        for (int i = 0; i < num_outputs; i++) {
            total += std::accumulate(confusion[i].begin(), confusion[i].end(), 0);
            correct += confusion[i][i];
        }

        double accuracy = (total > 0) ? (double)correct / total : 0.0;
        std::cout << "Accuracy: " << accuracy << std::endl;

        for (int c = 0; c < num_outputs; c++) {
            int TP = confusion[c][c];
            int FP = 0, FN = 0;

            for (int j = 0; j < num_outputs; j++) {
                if (j != c) {
                    FP += confusion[j][c];
                    FN += confusion[c][j];
                }
            }

            double precision = (TP + FP) ? (double)TP / (TP + FP) : 0.0;
            double recall    = (TP + FN) ? (double)TP / (TP + FN) : 0.0;
            double f1        = (precision + recall) ? 2 * (precision * recall) / (precision + recall) : 0.0;

            std::cout << "Class " << c 
                      << " -> Precision: " << precision 
                      << ", Recall: " << recall 
                      << ", F1-score: " << f1 
                      << std::endl;
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

void drawFigure(std::vector<double>& figure){
    int k = std::sqrt(figure.size());
    int tmp = 0;
    for (size_t i = 0; i < figure.size(); i++){
        if (figure[i] == 0) std::cout << "⬜";
        if (figure[i] == 1) std::cout << "⬛";
        tmp ++;
        if (tmp == k){
            tmp = 0;
            std::cout << std::endl;
        }
    }
}