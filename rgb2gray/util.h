#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

typedef struct
{
    int height;
    int width;
    int tot_pixels;
    unsigned char *pixel_buf; // RGB buffer
} Img;

Img *read_jpeg_img(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        perror("Unable to open file");
        return NULL;
    }

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    // Initialize decompression
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, fp);
    jpeg_read_header(&cinfo, TRUE);

    if (cinfo.num_components != 3)
    {
        fprintf(stderr, "Expected RGB image for grayscale conversion, but received %d channels.\n", cinfo.num_components);
        fclose(fp);
        return NULL;
    }

    jpeg_start_decompress(&cinfo);

    Img *img = (Img *)malloc(sizeof(Img));
    img->height = cinfo.image_height;
    img->width = cinfo.image_width;
    img->tot_pixels = img->height * img->width * 3;
    size_t row_size = cinfo.image_width * cinfo.num_components;

    img->pixel_buf = (unsigned char *)malloc(row_size * cinfo.image_height);
    if (img->pixel_buf == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for pixel buffer\n");
        free(img);
        fclose(fp);
        return NULL;
    }

    // Read the image scanlines
    while (cinfo.output_scanline < cinfo.image_height)
    {
        unsigned char *row_ptr = img->pixel_buf + cinfo.output_scanline * row_size;
        jpeg_read_scanlines(&cinfo, &row_ptr, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);

    return img;
}

void write_jpepg_img(const char *filename, int height, int width, int channels, unsigned char *gray_buf)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        perror("Unable to open output file");
        return;
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, fp);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = channels;
    if (channels == 1)
    {
        cinfo.in_color_space = JCS_GRAYSCALE;
    }
    else if (channels == 3)
    {
        cinfo.in_color_space = JCS_RGB;
    }
    else
    {
        fprintf(stderr, "Unsupported number of channels: %d\n", channels);
        jpeg_destroy_compress(&cinfo);
        fclose(fp);
        return;
    }

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 85, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height)
    {
        unsigned char *row_ptr = gray_buf + cinfo.next_scanline * width * channels;
        jpeg_write_scanlines(&cinfo, &row_ptr, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(fp);
}

#endif