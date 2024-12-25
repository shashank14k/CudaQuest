#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat load_hd_image(const char *filename)
{
    cv::Mat img = cv::imread(filename, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);

    if (img.empty())
    {
        std::cerr << "Couldn't open file" << filename << std::endl;
    }

    if (img.depth() == CV_8U)
    {
        img.convertTo(img, CV_32FC3, 1.0 / 255.0);
    }
    else if (img.depth() == CV_16U)
    {
        img.convertTo(img, CV_32FC3, 1.0 / 65535.0);
    }
    else if (img.depth() == CV_32F)
    {
        if (img.type() != CV_32FC3)
        {
            img.convertTo(img, CV_32FC3);
        }
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    return img;
}

void save_yuv_as_jpg(const float *yuv_buf, int rows, int cols, const char *out_path)
{
    cv::Mat yuv_image(rows, cols, CV_32FC3, const_cast<float *>(yuv_buf));
    cv::normalize(yuv_image, yuv_image, 0, 255, cv::NORM_MINMAX);
    yuv_image.convertTo(yuv_image, CV_8UC3);
    if (!cv::imwrite(out_path, yuv_image))
    {
        std::cerr << "Failed to save image: " << out_path << std::endl;
    }
}

void save_image(unsigned char *rgb_buf, int rows, int cols, const char *out_path)
{
    cv::Mat img(rows, cols, CV_8UC3, rgb_buf);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    if (cv::imwrite(out_path, img))
    {
        std::cout << "Image saved as " << out_path << std::endl;
    }
    else
    {
        std::cerr << "Error saving image: " << out_path << std::endl;
    }
}
