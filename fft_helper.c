#include "xcorr2.h"
#include <assert.h>
#include <string.h>
#include <fftw3.h>

complex double *dft_interpolate_2d(
        complex double *in,
        int height, int width,
        int scale_h, int scale_w,
        pthread_mutex_t *fftw_lock) {

    fftw_plan plan1, plan2;
    int out_height, out_width;
    complex double *in_fft;
    complex double *out_fft;
    complex double *out;

    assert(height >= 2 && TEST_2PWR(height));
    assert(width >= 2 && TEST_2PWR(width));
    assert(scale_w >= 1 && scale_h >= 1);

    out_height = height * scale_h;
    out_width = width * scale_w;
    
    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    in_fft = fftw_alloc_complex(height * width);
    out_fft = fftw_alloc_complex(out_height * out_width);
    out = fftw_alloc_complex(out_height * out_width);
    plan1 = fftw_plan_dft_2d(height, width, in, in_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    plan2 = fftw_plan_dft_2d(out_height, out_width, out_fft, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    fftw_execute(plan1);

    int y_offset = out_height - height;
    for (int y=0; y<out_height; y++) {
        int sy;
        complex double *out_fft_row = &out_fft[y*out_width];
        complex double *in_fft_row;

        if (y < height / 2)
            sy = y;
        else if (y >= out_height - height/2)
            sy = y - y_offset;
        else {
            memset(out_fft_row, 0, out_width * sizeof(complex double));
            continue;
        }

        in_fft_row = &in_fft[sy*width];

        memcpy(out_fft_row, in_fft_row, width/2 * sizeof(complex double));
        memset(&out_fft_row[width/2], 0, (out_width - width) * sizeof(complex double));
        memcpy(&out_fft_row[out_width - width/2], &in_fft_row[width/2], width/2 * sizeof(complex double));

        out_fft_row[out_width - width/2] /= 2;
        out_fft_row[width/2] = out_fft_row[out_width - width/2];
    }

    fftw_execute(plan2);

    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    fftw_free(in_fft);
    fftw_free(out_fft);
    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    // FIXME: remove this later
    // scale to match GMTSAR for debugging
    for (int i=0; i<out_height*out_width; i++)
        out[i] /= height * width;

    return out;
}

double *rdft_interpolate_2d(
        double *in,
        int height, int width,
        int scale_h, int scale_w,
        pthread_mutex_t *fftw_lock) {

    fftw_plan plan1, plan2;
    int out_height, out_width;
    complex double *in_fft;
    complex double *out_fft;
    double *out;

    assert(height >= 2 && TEST_2PWR(height));
    assert(width >= 2 && TEST_2PWR(width));
    assert(scale_w >= 1 && scale_h >= 1);

    out_height = height * scale_h;
    out_width = width * scale_w;
    
    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    in_fft = fftw_alloc_complex(height * (width/2 + 1));
    out_fft = fftw_alloc_complex(out_height * (out_width/2 + 1));
    out = fftw_alloc_real(out_height * out_width);
    plan1 = fftw_plan_dft_r2c_2d(height, width, in, in_fft, FFTW_ESTIMATE);
    plan2 = fftw_plan_dft_c2r_2d(out_height, out_width, out_fft, out, FFTW_ESTIMATE);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    fftw_execute(plan1);

    int y_offset = out_height - height;
    for (int y=0; y<out_height; y++) {
        int sy;
        complex double *out_fft_row = &out_fft[y*(out_width/2+1)];
        complex double *in_fft_row;

        if (y < height / 2)
            sy = y;
        else if (y >= out_height - height/2)
            sy = y - y_offset;
        else {
            memset(out_fft_row, 0, (out_width/2 + 1) * sizeof(complex double));
            continue;
        }

        in_fft_row = &in_fft[sy*(width/2+1)];
        memcpy(out_fft_row, in_fft_row, (width/2) * sizeof(complex double));
        out_fft_row[width/2] = in_fft_row[width/2] / 2;
        memset(&out_fft_row[width/2+1], 0, ((out_width - width)/2) * sizeof(complex double));
    }

    fftw_execute(plan2);

    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    fftw_free(in_fft);
    fftw_free(out_fft);
    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    // FIXME: remove this later
    // scale to match GMTSAR for debugging
    for (int i=0; i<out_height*out_width; i++)
        out[i] /= height * width;

    return out;
}
