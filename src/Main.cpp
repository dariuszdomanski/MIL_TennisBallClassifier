#include <cstdio>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "Logger.hpp"

#define UNUSED(x) ((void)(x))

#define CHECK(status) do {   \
    int res = (int)(status); \
    assert(res == 0);        \
    UNUSED(res);             \
} while(false)


std::vector<float> prepareImage(cv::Mat& img)
{
    using namespace cv;

    int c = 3;
    int h = 224;
    int w = 224;

    float scale = min(float(w)/img.cols,float(h)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat rgb;
    cv::cvtColor(img, rgb, CV_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized,scaleSize,0,0,INTER_CUBIC);

    cv::Mat cropped(h, w,CV_8UC3, 127);
    Rect rect((w- scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width,scaleSize.height); 
    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    if (c == 3)
        cropped.convertTo(img_float, CV_32FC3, 1/255.0);
    else
        cropped.convertTo(img_float, CV_32FC1 ,1/255.0);

    //HWC TO CHW
    std::vector<Mat> input_channels(c);
    cv::split(img_float, input_channels);

    std::vector<float> result(h*w*c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    return result;
}

auto main(int argc, char** argv) -> int
{
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    bool result = parser->parseFromFile("../models/resnet152/model.onnx", 1);
    nvinfer1::ICudaEngine* cudaEngine = builder->buildCudaEngine(*network);
    nvinfer1::IExecutionContext* executionContext =
        cudaEngine->createExecutionContext();

    printf("Number of bindings: %d\n", cudaEngine->getNbBindings());
    void* buffers[2];
    std::vector<float> output(1000);
    cv::Mat img = cv::imread("krowa.jpg");
    std::vector<float> input = prepareImage(img);

    CHECK(cudaMalloc(&buffers[0], input.size() * sizeof(float)));
    CHECK(cudaMalloc(&buffers[1], output.size() * sizeof(float)));

    CHECK(cudaMemcpy(buffers[0], input.data(),
        input.size() * sizeof(float),  cudaMemcpyHostToDevice));

    executionContext->execute(1, buffers);

    CHECK(cudaMemcpy(output.data(), buffers[1], output.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    
    int i = std::distance(output.begin(),
        std::max_element(output.begin(), output.end()));
    printf("Output: %d, probability: %f\n", i, output[i]);;
    return 0;
}
