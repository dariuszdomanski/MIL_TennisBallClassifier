#include <cstdio>
#include <fstream>
#include <iostream>

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <NvInfer.h>
#include <NvOnnxParserRuntime.h>
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>

#include "tensorrt/ImageLoader.hpp"
#include "tensorrt/Logger.hpp"


#define UNUSED(x) ((void)(x))

#define CHECK(status) do {   \
    int res = (int)(status); \
    assert(res == 0);        \
    UNUSED(res);             \
} while(false)


cv::Mat slMat2cvMat(sl::Mat& input)
{
    int cv_type = -1;
    switch (input.getDataType())
    {
        case sl::MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
        default: break;
    }

    return cv::Mat(input.getHeight(), input.getWidth(), cv_type,
        input.getPtr<sl::uchar1>(sl::MEM_CPU));
}

auto main(int argc, char** argv) -> int
{
    std::shared_ptr<bool> tennisBallDetection = std::make_shared<bool>(false);

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(tensorrt::Logger::getInstance());
    nvinfer1::INetworkDefinition* network = builder->createNetwork();
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network,
        tensorrt::Logger::getInstance());
    bool result = parser->parseFromFile("../models/shufflenet/model.onnx", 1);
    nvinfer1::ICudaEngine* cudaEngine = builder->buildCudaEngine(*network);
    nvinfer1::IExecutionContext* executionContext =
        cudaEngine->createExecutionContext();

    printf("Number of bindings: %d\n", cudaEngine->getNbBindings());
    void* buffers[2];
    std::vector<float> output(1000);

    sl::Camera zed;

    sl::InitParameters param;
    param.camera_resolution = sl::RESOLUTION_HD720;

    sl::ERROR_CODE err = zed.open(param);
    if (err != sl::SUCCESS)
    {
        std::cout << toString(err) << std::endl;
        zed.close();
        return 1;
    }

    sl::Mat zed_image;

    bool cudaMemAlloc = false;
    while (true)
    {
        if (zed.grab() != sl::SUCCESS)
        {
            std::cout<<"Capture read error"<<std::endl;
            return 0;
        }
        else
        {
            zed.retrieveImage(zed_image, sl::VIEW_LEFT);
            cv::Mat frame_in(slMat2cvMat(zed_image));
            cv::Mat frame_in_bgr;
            cv::cvtColor(frame_in, frame_in_bgr, cv::COLOR_RGBA2BGR);

            CUctx_st* cudaContext;
            cuCtxPopCurrent(&cudaContext);

            std::vector<float> input = std::move(tensorrt::load<3, 224, 224>(frame_in_bgr));
        
            if (cudaMemAlloc == false)
            {
                cudaMemAlloc = true;
                CHECK(cudaMalloc(&buffers[0], input.size() * sizeof(float)));
                CHECK(cudaMalloc(&buffers[1], output.size() * sizeof(float)));
            }

            CHECK(cudaMemcpy(buffers[0], input.data(),
                input.size() * sizeof(float),  cudaMemcpyHostToDevice));

            executionContext->execute(1, buffers);

            CHECK(cudaMemcpy(output.data(), buffers[1], output.size() * sizeof(float),
                cudaMemcpyDeviceToHost));
            
            cuCtxPushCurrent(cudaContext);

            int i = std::distance(output.begin(), std::max_element(output.begin(), output.end()));

            if (i == 852 && *tennisBallDetection == false)
            {
                printf("TennisBall detected, probability: %f%% \n", output[i] * 100);
                *tennisBallDetection = true;
            }
            else if (*tennisBallDetection == true)
            {
                printf("TennisBall not detected\n");
                *tennisBallDetection = false;
            }
        } 
    }

    return 0;
}
