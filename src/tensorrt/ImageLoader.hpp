#ifndef TENSORRT_IMAGELOADER_HPP_
#define TENSORRT_IMAGELOADER_HPP_

#include <vector>
#include <opencv2/opencv.hpp>


namespace tensorrt
{

template <const int c, const int h, const int w>
std::vector<float> load(const cv::Mat& img)
{
    float scale = cv::min(static_cast<float>(w)/img.cols, static_cast<float>(h)/img.rows);
    auto scaleSize = cv::Size(img.cols*scale, img.rows*scale);

    cv::Mat rgb;
    cv::cvtColor(img, rgb, CV_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, scaleSize, 0, 0, CV_INTER_CUBIC);

    cv::Mat cropped(h, w, CV_8UC3, 127);
    cv::Rect rect((w-scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width, scaleSize.height); 
    resized.copyTo(cropped(rect));

    cv::Mat imgFloat;
    if (c == 3)
        cropped.convertTo(imgFloat, CV_32FC3, 1/255.0);
    else
        cropped.convertTo(imgFloat, CV_32FC1 ,1/255.0);

    //HWC TO CHW
    std::vector<cv::Mat> inputChannels(c);
    cv::split(imgFloat, inputChannels);

    std::vector<float> result(h*w*c);
    auto data = result.data();
    constexpr int channelLength = h * w;
    for (int i = 0; i < c; i++)
    {
        memcpy(data, inputChannels[i].data, channelLength*sizeof(float));
        data += channelLength;
    }

    return result;
}

}  // tensorrt

#endif  // TENSORRT_IMAGELOADER_HPP_
