#ifndef LOGGER_HPP_
#define LOGGER_HPP_

#include <cstdio>
#include <NvInfer.h>


class Logger : public nvinfer1::ILogger           
{
private:
    using Severity = nvinfer1::ILogger::Severity;

    void log(Severity severity, const char* msg) override
    {
        if (severity != Severity::kINFO)
            printf("%s\n", msg);
    }
} gLogger;

#endif  // LOGGER_HPP_