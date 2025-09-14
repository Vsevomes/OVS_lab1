#include <iostream>
#include <sstream>
#include <algorithm>
#include "NeuralNet.h"

#define TARGET_ERROR 0.01

struct Sample {
    std::vector<double> input;
    std::vector<double> target;
};

std::vector<Sample> loadSamples(const std::string& filename);
std::vector<double> loadShape(const std::string& filename);
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

        std::string train_file = argv[2];
        int num_inputs = std::stoi(argv[3]);
        int hidden_layers = std::stoi(argv[4]);
        int neurons_per_layer = std::stoi(argv[5]);
        int num_outputs = std::stoi(argv[6]);

        NeuralNet net(num_inputs, hidden_layers, neurons_per_layer, num_outputs);
        
        std::vector<Sample> samples = loadSamples(train_file);
        if (samples.empty()) {
            std::cerr << "No test data" << std::endl;
            return 1;
        }

        // training settings
        double err = 1.0;         
        int epoch = 0;
        int maxEpochs = 100000; 

        // training process
        while (err > TARGET_ERROR && epoch < maxEpochs) {
            err = 0.0;
            for (auto &s : samples) {
                net.train(s.input, s.target);
                std::vector<double> out = net.forward(s.input);
                for (int i = 0; i < 3; i++)
                    err += (s.target[i] - out[i]) * (s.target[i] - out[i]);
            }
            err /= samples.size();
            epoch++;
            if (epoch % 1000 == 0) {
                std::cout << "Epoch " << epoch << ", ERR = " << err << std::endl;
            }
        }

        std::cout << "Learning completed for " << epoch << " epochs, ERR = " << err << std::endl;

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

        std::vector<double> input = loadShape(input_file);
        if (input.empty()) {
            std::cerr << "Error: empty data" << std::endl;
            return 1;
        }

        // start
        std::vector<double> out = net.forward(input);

        int predicted = std::max_element(out.begin(), out.end()) - out.begin();

        if (out[0] == out[predicted] && out[predicted] > 0.5) {
            std::cout << "Shape: circle" << std::endl << "Accuracy: " << out[predicted] << std::endl;
        }
        else if (out[1] == out[predicted] && out[predicted] > 0.5) {
            std::cout << "Shape: rectangle" << std::endl << "Accuracy: " << out[predicted] << std::endl;
        }
        else if (out[2] == out[predicted] && out[predicted] > 0.5) {
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

std::vector<Sample> loadSamples(const std::string& filename) {
    std::vector<Sample> samples;
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Erro open test file" << filename << std::endl;
        return samples;
    }

    int num_shapes, num_pixels;
    file >> num_shapes >> num_pixels;
    std::string line;
    getline(file, line);

    int side = static_cast<int>(sqrt(num_pixels));
    if (side * side != num_pixels) {
        std::cerr << "Wrong pixel size" << std::endl;
        return samples;
    }

    for (int s = 0; s < num_shapes; s++) {
        Sample sample;
        sample.input.reserve(num_pixels);

        int rowsRead = 0;
        while (rowsRead < side && getline(file, line)) {
            if (line.empty()) continue;
            for (char c : line) {
                if (c == '0' || c == '1')
                    sample.input.push_back(c - '0');
            }
            rowsRead++;
        }

        if (sample.input.size() != num_pixels) {
            std::cerr << "Wrong number of points in shape" << std::endl;
            continue;
        }

        sample.target = std::vector<double>(num_shapes, 0.0);
        if (s >= 0 && s < num_shapes) sample.target[s] = 1.0;

        samples.push_back(sample);
    }

    file.close();
    return samples;
}

std::vector<double> loadShape(const std::string& filename) {
    std::vector<double> input;
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Ошибка открытия файла " << filename << std::endl;
        return input;
    }

    std::string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        for (char c : line) {
            if (c == '0' || c == '1') {
                input.push_back(c - '0');
            }
        }
    }

    file.close();
    return input;
}

void menu(std::string prog_name){
    std::cerr << "Usage:" << std::endl;
        std::cerr << "  For train model: " 
             << prog_name << " train <train_file> <inputs> <hiddenLayers> <neuronsPerLayer> <outputs>" << std::endl;
        std::cerr << "  For use model: " 
             << prog_name << " predict <model_file> <input_file>" << std::endl;
}