#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <time.h>



#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"



typedef struct {
    uint32_t *pixels;
    int width, height, stride;
} Img;


#define IMG_AT(img, row, col) (img).pixels[(row)*(img).stride + (col)]

typedef struct {
    float *items;
    int width, height, stride;
} Mat;

#define MAT_AT(mat, row, col) (mat).items[(row)*(mat).stride + (col)]
#define MAT_WITHIN(mat, row, col) \
    (0 <= (col) && (col) < (mat).width && 0 <= (row) && (row) < (mat).height)

static Mat mat_alloc(int width, int height)
{
    Mat mat = {0};
    mat.items = malloc(sizeof(float)*width*height);
    assert(mat.items != NULL);
    mat.width = width;
    mat.height = height;
    mat.stride = width;
    return mat;
}

// https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
static float rgb_to_lum(uint32_t rgb)
{
    float r = ((rgb >> (8*0)) & 0xFF)/255.0;
    float g = ((rgb >> (8*1)) & 0xFF)/255.0;
    float b = ((rgb >> (8*2)) & 0xFF)/255.0;
    return 0.2126*r + 0.7152*g + 0.0722*b;
}

static void luminance(Img img, Mat lum)
{
    assert(img.width == lum.width);
    assert(img.height == lum.height);
    for (int y = 0; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            MAT_AT(lum, y, x) = rgb_to_lum(IMG_AT(img, y, x));
        }
    }
}

static float sobel_filter_at(Mat mat, int cx, int cy)
{
    static float gx[3][3] = {
        {1.0, 0.0, -1.0},
        {2.0, 0.0, -2.0},
        {1.0, 0.0, -1.0},
    };

    static float gy[3][3] = {
        {1.0, 2.0, 1.0},
        {0.0, 0.0, 0.0},
        {-1.0, -2.0, -1.0},
    };

    float sx = 0.0;
    float sy = 0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int x = cx + dx;
            int y = cy + dy;
            float c = MAT_WITHIN(mat, y, x) ? MAT_AT(mat, y, x) : 0.0;
            sx += c*gx[dy + 1][dx + 1];
            sy += c*gy[dy + 1][dx + 1];
        }
    }

    return sx*sx + sy*sy;
}

static void sobel_filter(Mat mat, Mat grad)
{
    assert(mat.width == grad.width);
    assert(mat.height == grad.height);

    for (int cy = 0; cy < mat.height; ++cy) {
        for (int cx = 0; cx < mat.width; ++cx) {
            MAT_AT(grad, cy, cx) = sobel_filter_at(mat, cx, cy);
        }
    }
}

static void usage(const char *program)
{
    fprintf(stderr, "Usage: %s <input> <output>\n", program);
}

static void min_and_max(Mat mat, float *mn, float *mx)
{
    *mn = FLT_MAX;
    *mx = FLT_MIN;
    for (int y = 0; y < mat.height; ++y) {
        for (int x = 0; x < mat.width; ++x) {
            float value = MAT_AT(mat, y, x);
            if (value < *mn) *mn = value;
            if (value > *mx) *mx = value;
        }
    }
}

static bool dump_mat(const char *file_path, Mat mat)
{
    float mn, mx;
    min_and_max(mat, &mn, &mx);

    uint32_t *pixels = NULL;
    bool result = true;

    pixels = malloc(sizeof(*pixels)*mat.width*mat.height);
    assert(pixels != NULL);

    for (int y = 0; y < mat.height; ++y) {
        for (int x = 0; x < mat.width; ++x) {
            int i = y*mat.width + x;
            float t = (MAT_AT(mat, y, x) - mn)/(mx - mn);
            uint32_t value = 255*t;
            pixels[i] = 0xFF000000|(value<<(8*2))|(value<<(8*1))|(value<<(8*0));
        }
    }

    if (!stbi_write_png(file_path, mat.width, mat.height, 4, pixels, mat.width*sizeof(*pixels))) {
        fprintf(stderr, "ERROR: could not save file %s", file_path);
    }

    printf("OK: generated %s\n", file_path);

    free(pixels);
    return result;
}

int main(int argc, char **argv){
    const char *file_path = "input.jpg";
    const char *out_file_path = "output.jpg";

    int width_, height_;
    uint32_t *pixels_ = (uint32_t*)stbi_load(file_path, &width_, &height_, NULL, 4);
    if (pixels_ == NULL) {
        fprintf(stderr, "ERROR: could not read %s\n", file_path);
        return 1;
    }
    Img img = {
        .pixels = pixels_,
        .width = width_,
        .height = height_,
        .stride = width_,
    };

    Mat lum = mat_alloc(width_, height_);
    Mat grad = mat_alloc(width_, height_);

    clock_t start_luminance = clock();
    luminance(img, lum);
    clock_t end_luminance = clock();
    double time_luminance = (double)(end_luminance - start_luminance) / CLOCKS_PER_SEC;
    printf("Tempo di esecuzione di luminance(): %f secondi\n", time_luminance);

    // Timer per sobel_filter
    clock_t start_sobel = clock();
    sobel_filter(lum, grad);
    clock_t end_sobel = clock();
    double time_sobel = (double)(end_sobel - start_sobel) / CLOCKS_PER_SEC;
    printf("Tempo di esecuzione di sobel_filter(): %f secondi\n", time_sobel);

    clock_t start_dump_lum = clock();
    dump_mat("lum.png", lum);
    clock_t end_dump_lum = clock();
    double time_dump_lum = (double)(end_dump_lum - start_dump_lum) / CLOCKS_PER_SEC;
    printf("Tempo di esecuzione di dump_mat(lum.png): %f secondi\n", time_dump_lum);

    clock_t start_dump_grad = clock();
    dump_mat("grad.png", grad);
    clock_t end_dump_grad = clock();
    double time_dump_grad = (double)(end_dump_grad - start_dump_grad) / CLOCKS_PER_SEC;
    printf("Tempo di esecuzione di dump_mat(grad.png): %f secondi\n", time_dump_grad);

    printf("OK: generated %s\n", out_file_path);

    return 0;
}
