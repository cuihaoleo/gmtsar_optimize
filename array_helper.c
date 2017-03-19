#include "xcorr2.h"
#include <assert.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <fftw3.h>

void c64_array_print(const char *fmt, complex double *arr, int n, int m) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            printf(fmt, creal(arr[i*m+j]), cimag(arr[i*m+j]));
        }
        putchar('\n');
    }
}

complex double *c64_array_slice(
        const complex double *mat, int n_cols,
        int tl_y, int s_y,
        int tl_x, int s_x) {

    assert(tl_y >= 0 && tl_x >= 0 && tl_x < n_cols);

    complex double *arr;
    arr = fftw_alloc_complex(s_x * s_y);

    for (int i=0; i<s_y; i++)
        memcpy(&arr[i*s_x], &mat[(i+tl_y)*n_cols + tl_x], s_x * sizeof(complex double));

    return arr;
}

double *f64_array_slice(
        const double *mat, int n_cols,
        int tl_y, int s_y,
        int tl_x, int s_x) {

    assert(tl_y >= 0 && tl_x >= 0 && tl_x < n_cols);

    double *arr;
    arr = fftw_alloc_real(s_x * s_y);

    for (int i=0; i<s_y; i++)
        memcpy(&arr[i*s_x], &mat[(i+tl_y)*n_cols + tl_x], s_x * sizeof(double));

    return arr;
}

void f64_array_stats(
        const double *array,
        int ny, int nx,
        double *average, double *max,
        int *argmax_y, int *argmax_x) {
    int xm, ym;
    double total, maxval;

    total = 0;
    xm = ym = INT_MIN;
    maxval = -INFINITY;

    for (int i=0, k=0; i<ny; i++)
        for (int j=0; j<nx; j++, k++) {
            total += array[k];

            if (array[k] > maxval) {
                maxval = array[k];
                ym = i;
                xm = j;
            }
        }

    if (average) *average = total / (nx * ny);
    if (max) *max = maxval;
    if (argmax_y) *argmax_y = ym;
    if (argmax_x) *argmax_x = xm;
}
