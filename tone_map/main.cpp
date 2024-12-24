#include "hdimage.h"
#include "transforms.cuh"
#include <opencv2/core/core.hpp>

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <input-image-path> <output-image-path>\n", argv[0]);
        return -1;
    }

    cv::Mat image = load_hd_image(argv[1]);
    float *yuv_buf = convert_rgb_to_yuv(image.ptr<float>(), image.rows, image.cols, 3);
    save_yuv_as_jpg(yuv_buf, image.rows, image.cols, argv[2]);
    get_image_histogram(image.ptr<float>(), image.rows, image.cols, 3, 255);
}