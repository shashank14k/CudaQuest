#include <cuda_runtime.h>
#define SHARED_MEM_SIZE 256
#define MAX_THREADS_PER_BLOCK 1024
__global__ void rgb_to_yuv(float *in_buf, float *out_buf, int tot_pixels);
__global__ void find_minimum(float *buf, float *buf_r, int stride, int buf_size);
__global__ void find_maximum(float *buf, float *buf_r, int stride, int buf_size);
__global__ void compute_image_histogram_naive(const float *img, const int img_size, float min_bin_val, float max_bin_val, int *d_bins, const int numBins);
__global__ void compute_image_histogram_faster(const float *img, const int img_size, const int stride, float min_bin_val, float max_bin_val, int *d_bins, const int numBins);
__global__ void compute_histogram_distribution(const int *dbins, int *cdf, int numBins);
extern "C" float *convert_rgb_to_yuv(float *buf, int rows, int cols, int channels);
extern "C" void *get_image_histogram(float *buf, int rows, int cols, int channels, int n_bins);