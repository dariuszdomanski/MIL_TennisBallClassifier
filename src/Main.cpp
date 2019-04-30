#include <cstdio>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "Logger.hpp"


auto main(int argc, char** argv) -> int
{
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    bool result = parser->parseFromFile("../models/tiny_yolov2/model.onnx", 1);
    return 0;
}
