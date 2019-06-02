#include <cstdio>
#include <fstream>
#include <iostream>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include "Logger.hpp"
#include "Utils.hpp"


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

cv::Mat slMat2cvMat(sl::Mat& input) {
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

std::vector<std::string> split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

std::vector<std::string> readLabels(std::string filename)
{
    std::ifstream infile(filename);
    std::string label;
    std::vector<std::string> labels;
    while (std::getline(infile, label))
    {
        auto tokens = split(label, "'");
        std::cout << tokens[1] << std::endl;
        labels.push_back(label);
    }
    return labels;
}

auto main(int argc, char** argv) -> int
{
    auto labels = readLabels("../res/imagenet.txt");
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    bool result = parser->parseFromFile("../models/shufflenet/model.onnx", 1);
    nvinfer1::ICudaEngine* cudaEngine = builder->buildCudaEngine(*network);
    nvinfer1::IExecutionContext* executionContext =
        cudaEngine->createExecutionContext();

    printf("Number of bindings: %d\n", cudaEngine->getNbBindings());
    void* buffers[2];
    std::vector<float> output(1000);

    const char* gst =  "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)NV12, framerate=(fraction)24/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

    cv::VideoCapture cap(gst);

    if(!cap.isOpened())
    {
        std::cout<<"Failed to open camera."<<std::endl;
        return -1;
    }

    unsigned int width = cap.get(CV_CAP_PROP_FRAME_WIDTH); 
    unsigned int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    unsigned int pixels = width*height;
    cv::Mat frame_in(width, height, cap.get(CV_CAP_PROP_FORMAT));

    cv::namedWindow("SztywnyKlasyfikator", CV_WINDOW_AUTOSIZE);

    bool cudaMemAlloc = false;
    while (true)
    {
        if (!cap.read(frame_in))
        {
            std::cout<<"Capture read error"<<std::endl;
            return 0;
        }
        else
        {
            std::vector<float> input = prepareImage(frame_in);

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
            
            int i = std::distance(output.begin(),
                std::max_element(output.begin(), output.end()));
            printf("Output: %s probability: %f%% \n", labels[i].c_str(), output[i] * 100);
            cv::imshow("SztywnyKlasyfikator", frame_in);
            cv::waitKey(1); // let imshow draw and wait for next frame 8 ms for 120 fps
        }       
    }

    cap.release();

    network->destroy();
    builder->destroy();
    executionContext->destroy();
    cudaEngine->destroy();

    return 0;
}
