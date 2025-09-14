#include <iostream>
#include <sstream>
#include "NeuralNet.h"

#define TARGET_ERROR 0.001

struct Sample {
    std::vector<double> input;  // 49 значений
    std::vector<double> target; // one-hot вектор [1,0,0] или [0,1,0] или [0,0,1]
};

std::vector<Sample> loadSamples(const std::string& filename);

int main(int argc, char* argv[]) {
    srand(time(NULL));

    // создаём сеть: 2 входа, 1 скрытый слой по 2 нейрона, 1 выход, learning rate = 0.1
    NeuralNet net(2, 3, 3, 1);

    std::vector<std::vector<double>> inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
    std::vector<std::vector<double>> targets = {{0}, {0}, {0}, {1}};

    // параметры обучения по ошибке
    double err = 1.0;            // начальная ошибка
    int epoch = 0;
    int maxEpochs = 100000;      // защита от бесконечного цикла

    // цикл обучения до достижения targetError
    while (err > TARGET_ERROR && epoch < maxEpochs) {
        err = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            net.train(inputs[i], targets[i]);
            std::vector<double> out = net.forward(inputs[i]);
            err += (targets[i][0] - out[0])*(targets[i][0] - out[0]);
        }
        err = err / 2; // средняя ошибка
        epoch++;

        // вывод прогресса каждые 1000 эпох
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", MSE = " << err << std::endl;
        }
    }

    std::cout << "Обучение завершено за " << epoch << " эпох, MSE = " << err << std::endl;

    std::string file = "../data/model";

    // сохраняем обученную модель
    net.save(file);

    // создаём новую сеть и загружаем модель
    NeuralNet net2(file);

    // проверка работы сети
    std::cout << "Результаты после загрузки модели:" << std::endl;
    for (int i = 0; i < inputs.size(); i++) {
        std::vector<double> out = net2.forward(inputs[i]);
        std::cout << inputs[i][0] << " AND " << inputs[i][1]
             << " = " << out[0] << std::endl;
    }

    return 0;
}

// // Считываем данные из файла
// std::vector<Sample> loadSamples(const std::string& filename) {
//     std::vector<Sample> samples;
//     std::ifstream file(filename);
//     if (!file) {
//         std::cerr << "Ошибка открытия файла " << filename << std::endl;
//         return samples;
//     }

//     int num_shapes, num_pixels;
//     file >> num_shapes >> num_pixels;
//     std::string line;
//     getline(file, line); // считываем остаток первой строки

//     for (int s = 0; s < num_shapes; s++) {
//         Sample sample;
//         sample.input.reserve(num_pixels);

//         for (int row = 0; row < 7; row++) {
//             getline(file, line);
//             if (line.empty()) {
//                 row--; // пропускаем пустые строки
//                 continue;
//             }
//             for (char c : line) {
//                 if (c == '0' || c == '1')
//                     sample.input.push_back(c - '0');
//             }
//         }

//         // создаём one-hot вектор для цели
//         sample.target = {0,0,0};
//         if (s >= 0 && s <= 2) sample.target[s] = 1;

//         samples.push_back(sample);
//     }

//     file.close();
//     return samples;
// }
