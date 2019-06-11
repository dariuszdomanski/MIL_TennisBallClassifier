#ifndef TENSORRT_LOGGER_HPP_
#define TENSORRT_LOGGER_HPP_

#include <cstdio>
#include <NvInfer.h>


namespace tensorrt
{

class Logger : public nvinfer1::ILogger           
{
public:
    static Logger& getInstance()
    {
        static Logger instance;
        return instance;
    }

private:
    using Severity = nvinfer1::ILogger::Severity;

    void log(Severity severity, const char* msg) override
    {
        printf("[TENSORRT] %s\n", msg);
    }
};

}  // tensorrt

#endif  // TENSORRT_LOGGER_HPP_