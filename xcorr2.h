#ifndef __XCORR2_H_INCLUDED__
#define __XCORR2_H_INCLUDED__

#include <stdio.h>
#include <stdbool.h>
#include <complex.h>
#include <pthread.h>
#include <glib.h>

#define TEST_2PWR(n) ((n) > 0 && ((n) & ((n) - 1)) == 0)

// array_helper.c
void c64_array_print(const char *fmt, complex double *arr, int n, int m);
complex double *c64_array_slice(const complex double *mat, int n_cols,
                                int tl_y, int s_y, int tl_x, int s_x);
double *f64_array_slice(const double *mat, int n_cols,
                        int tl_y, int s_y, int tl_x, int s_x);

void f64_array_stats(const double *array, int ny, int nx,
                     double *average, double *max,
                     int *argmax_y, int *argmax_x);

// fft_helper.c
complex double *dft_interpolate_2d(complex double *in, int height, int width,
                                   int scale_h, int scale_w,
                                   pthread_mutex_t *fftw_lock);
double *rdft_interpolate_2d(double *in, int height, int width,
                            int scale_h, int scale_w,
                            pthread_mutex_t *fftw_lock);

// prm_helper.c
struct prm_handler {
    GHashTable *entry;
};

struct prm_handler prm_open(const char *fname);
void prm_close(struct prm_handler *handler);
const char *prm_get_str(const struct prm_handler handler, const char *key);
int prm_get_int(const struct prm_handler handler, const char *key);
double prm_get_f64(const struct prm_handler handler, const char *key);
#endif
