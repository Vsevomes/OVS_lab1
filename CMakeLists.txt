cmake_minimum_required(VERSION 3.16) 
project(neural_net LANGUAGES CXX)    

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

set(SOURCES
    main.cpp
    src/NeuralNet.cpp
    src/NeuralLayer.cpp
    src/Neuron.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

# target_link_libraries(${PROJECT_NAME} PRIVATE some_library)
