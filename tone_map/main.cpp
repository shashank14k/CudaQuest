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
    unsigned char *equalized_rgb_img = equalize_histogram(image.ptr<float>(), image.rows, image.cols, 3, 256);
    save_image(equalized_rgb_img, image.rows, image.cols, argv[2]);
    free(equalized_rgb_img);
}