#include <cstdio>
#include <iostream>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <sl/Camera.hpp>

#include "Logger.hpp"
#include "Utils.hpp"


using namespace sl;

auto main(int argc, char** argv) -> int
{
    return 0;
}

// auto main(int argc, char** argv) -> int
// {
//     // Create a ZED camera
//     Camera zed;

//     // Set configuration parameters for the ZED
//     InitParameters initParameters;
//     initParameters.camera_resolution = RESOLUTION_HD720;
//     initParameters.depth_mode = DEPTH_MODE_NONE;
// 	initParameters.sdk_verbose = true;

//     // Open the camera
//     ERROR_CODE err = zed.open(initParameters);
//     if (err != SUCCESS)
//     {
//         std::cout << toString(err) << std::endl;
//         zed.close();
//         return -1; // Quit if an error occurred
//     }

//     sl::StreamingParameters stream_params;
//     stream_params.codec = sl::STREAMING_CODEC_AVCHD;
//     stream_params.bitrate = 8000;
//     err = zed.enableStreaming(stream_params);
//     if (err != SUCCESS)
//     {
//         std::cout << "Streaming initialization error. " << toString(err) << std::endl;
//         zed.close();
//         return -2;
//     }

//     setCtrlHandler();

//     int fc = 0;
//     while (!exit_app)
//     {
//         if (zed.grab() == SUCCESS)
//         {
//             sl::sleep_ms(1);
//             fc++;
//         }
//     }

//     // Stop recording
//     zed.disableStreaming();
//     zed.close();
//     return 0;
// }

// auto main(int argc, char** argv) -> int
// {
//     nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
//     nvinfer1::INetworkDefinition* network = builder->createNetwork();
//     nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
//     // bool result = parser->parseFromFile("../models/tiny_yolov2/model.onnx", 0);  // ERROR on  TX2, work well on Desktop
//     return 0;
// }