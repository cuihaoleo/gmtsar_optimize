#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <complex.h>
#include <pthread.h>
#include <fftw3.h>
#include <glib.h>
#include <unistd.h>
#include <stdbool.h>

#define TEST_2PWR(n) ((n) > 0 && ((n) & ((n) - 1)) == 0)

struct st_xcorr {
    int m_nx, m_ny;
    int s_nx, s_ny;

    int x_offset, y_offset;
    int xsearch, ysearch;
    int nxl, nyl;
    int astretcha;

    int ri;
    int interp_factor;
    int n2x, n2y;  // high-res correlation window

    bool interp_flag;

    char *m_path;
    char *s_path;
};

struct st_corr_thread_data {
    const struct st_xcorr *xc;
    complex double *c1;
    complex double *c2;
    double xoff, yoff;
    int loc_x, loc_y;
    double corr;
};

void print_complex_double_array(const char *fmt, complex double *arr, int n, int m) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            printf(fmt, creal(arr[i*m+j]), cimag(arr[i*m+j]));
        }
        putchar('\n');
    }
}

complex double *load_slc_rows(FILE *fin, int start, int n_rows, int nx) {
    long offset;
    short *tmp;
    complex double *arr;

    offset = nx * start * sizeof(short) * 2;
    fseek(fin, offset, SEEK_SET);

    tmp = malloc(nx * sizeof(short) * 2);
    arr = fftw_alloc_complex(n_rows * nx);

    for (int i=0; i<n_rows; i++) {
        if (fread(tmp, 2*sizeof(short), nx, fin) != (unsigned long)nx) {
            perror("Failed to read data from SLC file!");
            exit(-1);
        }

        for (int j=0; j<nx; j++)
            arr[i*nx + j] = tmp[2*j] + tmp[2*j+1] * I;
    }

    free(tmp);
    return arr;
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

complex double *dft_interpolate(
        complex double *in,
        int length,
        int scale,
        pthread_mutex_t *fftw_lock) {

    fftw_plan plan1, plan2;
    complex double *in_fft;
    complex double *out;

    assert(length >= 2 && TEST_2PWR(length));
    assert(scale >= 2 && TEST_2PWR(scale));
    
    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    in_fft = fftw_alloc_complex(length * scale);
    out = fftw_alloc_complex(length * scale);
    plan1 = fftw_plan_dft_1d(length, in, in_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    plan2 = fftw_plan_dft_1d(length * scale, in_fft, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    fftw_execute(plan1);

    int tail_offset = (scale - 1) * length;
    memcpy(&in_fft[length/2 + tail_offset], &in_fft[length/2], length/2 * sizeof(complex double));
    memset(&in_fft[length/2], 0, tail_offset * sizeof(complex double));

    fftw_execute(plan2);

    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    fftw_free(in_fft);
    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    for (int i=0; i<scale*length; i++)
        out[i] /= length;

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
    assert(scale_h >= 2 && TEST_2PWR(scale_h));
    assert(width >= 2 && TEST_2PWR(width));
    assert(scale_w >= 2 && TEST_2PWR(scale_w));

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
        complex double *in_fft_row = &in_fft[sy*(width/2+1)];

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

long double time_corr(
        complex double *c1,
        complex double *c2,
        int xsearch, int ysearch,
        int xoff, int yoff) {

    int nx_corr, ny_corr;
    int nx_win, ny_win;

    nx_corr = xsearch * 2;
    nx_win = nx_corr * 2;
    ny_corr = ysearch * 2;
    ny_win = ny_corr * 2;

    long double num, denom, denom1, denom2, result;

    num = denom1 = denom2 = 0.0;
    for (int i=0; i<ny_corr; i++)
        for (int j=0; j<nx_corr; j++) {
            long double a = c1[(ysearch + i + yoff) * nx_win + (xsearch + j + xoff)];
            long double b = c2[(ysearch + i) * nx_win + (xsearch + j)];

            num += a * b;
            denom1 += a * a;
            denom2 += b * b;
        }

    denom = sqrtl(denom1 * denom2);

    if (denom == 0.0) {
        fprintf(stderr, "calc_corr: denominator = zero: setting corr to 0 \n");
        result = 0.0;
    } else
        result = 100.0 * fabsl(num / denom);

    return result;
}

complex double *freq_corr(
        complex double *c1,
        complex double *c2,
        int nx_win, int ny_win,
        pthread_mutex_t *fftw_lock) {
    complex double *c1_fft, *c2_fft, *c3;
    fftw_plan plan1, plan2, plan3;

    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    c1_fft = fftw_alloc_complex(nx_win * ny_win);
    c2_fft = fftw_alloc_complex(nx_win * ny_win);
    c3 = fftw_alloc_complex(nx_win * ny_win);

    plan1 = fftw_plan_dft_2d(ny_win, nx_win, c1, c1_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    plan2 = fftw_plan_dft_2d(ny_win, nx_win, c2, c2_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    plan3 = fftw_plan_dft_2d(ny_win, nx_win, c1_fft, c3, FFTW_BACKWARD, FFTW_ESTIMATE);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    fftw_execute(plan1);
    fftw_execute(plan2);

    int isign = 1;
    for (int i=0; i<ny_win; i++) {
        for (int j=0; j<nx_win; j++) {
            c1_fft[i*nx_win + j] *= isign * conj(c2_fft[i*nx_win + j]);
            isign = -isign;
        }

        isign = -isign;
    }

    fftw_execute(plan3);

    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    fftw_free(c1_fft);
    fftw_free(c2_fft);
    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);
    fftw_destroy_plan(plan3);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    // FIXME: remove this later
    // scale to match GMTSAR for debugging
    for (int i=0; i<nx_win*ny_win; i++)
        c3[i] /= nx_win * ny_win;

    return c3;
}

void corr_thread(gpointer arg, gpointer user_data) {
    struct st_corr_thread_data *data = arg;
    pthread_mutex_t *lock = user_data;

    double mean1, mean2;
    int xsearch, ysearch;
    int nx_corr, ny_corr;
    int nx_win, ny_win;
    complex double *c1, *c2;

    xsearch = data->xc->xsearch;
    nx_corr = xsearch * 2;
    nx_win = nx_corr * 2;
    ysearch = data->xc->ysearch;
    ny_corr = ysearch * 2;
    ny_win = ny_corr * 2;

    c1 = data->c1;
    c2 = data->c2;

    // last part of assign_values
    if (data->xc->ri > 1) {
        for (int i=0; i<ny_win; i++) {
            complex double *interp1 = dft_interpolate(c1 + i*nx_win, nx_win, data->xc->ri, lock);
            complex double *interp2 = dft_interpolate(c2 + i*nx_win, nx_win, data->xc->ri, lock);
            int offset = data->xc->ri * nx_win / 2 - nx_win / 2;

            memcpy(&c1[i*nx_win], interp1 + offset, nx_win * sizeof(complex double));
            memcpy(&c2[i*nx_win], interp2 + offset, nx_win * sizeof(complex double));

            if (lock) pthread_mutex_lock(lock);
            fftw_free(interp1);
            fftw_free(interp2);
            if (lock) pthread_mutex_unlock(lock);
        }
    }

    mean1 = mean2 = 0.0;
    for (int i=0; i<nx_win*ny_win; i++) {
        c1[i] = cabs(c1[i]);
        c2[i] = cabs(c2[i]);

        mean1 += creal(c1[i]);
        mean2 += creal(c2[i]);
    }

    mean1 /= nx_win * ny_win;
    mean2 /= nx_win * ny_win;

    for (int i=0; i<nx_win*ny_win; i++) {
        c1[i] -= mean1;
        c2[i] -= mean2;
    }

    // make_mask and mask
    for (int i=0; i<ny_win; i++)
        for (int j=0; j<nx_win; j++) {
            if (i < ysearch
                    || i >= ny_win - ysearch
                    || j < xsearch
                    || j >= nx_win - xsearch)
                c2[i*nx_win + j] = 0;
        }


    // calc correlation with 2D FFT
    complex double *c3, *corr_c;
    c3 = freq_corr(c1, c2, nx_win, ny_win, lock);
    corr_c = c64_array_slice(c3, nx_win, ysearch, ny_corr, xsearch, nx_corr);

    double *corr;
    corr = fftw_alloc_real(nx_corr * ny_corr);
    for (int k=0; k<nx_corr*ny_corr; k++)
        corr[k] = cabs(corr_c[k]);

    //puts("ARRAY corr:");
    //print_complex_double("%+04.2f%+04.2fj\t", corr, ny_corr, nx_corr);

    int xpeak, ypeak;
    double cmax, cave, max_corr;

    f64_array_stats(corr, ny_corr, nx_corr, &cave, &cmax, &ypeak, &xpeak);
    xpeak -= xsearch;
    ypeak -= ysearch;

    cave /= nx_corr * ny_corr;
    max_corr = time_corr(c1, c2, xsearch, ysearch, xpeak, ypeak);

    //fprintf(stderr, "xypeak: (%d, %d)\n", xpeak, ypeak);
    //fprintf(stderr, "max_corr: %g\n", cmax);

    double xfrac = 0.0, yfrac = 0.0;

    // high-res correlation
    if (data->xc->interp_flag) {
        int factor = data->xc->interp_factor;
        int nx_corr2 = data->xc->n2x;
        int ny_corr2 = data->xc->n2y;
        double *corr2;
        double *hi_corr;

        assert(nx_corr2 >= 2 && TEST_2PWR(nx_corr2));
        assert(ny_corr2 >= 2 && TEST_2PWR(ny_corr2));

        // FIXME: remove this later
        // scale to match GMTSAR for debugging
        for (int k=0; k<nx_corr*ny_corr; k++)
            corr[k] *= max_corr / cmax;

        // FIXME: original GMTSAR are vulnerable to memory violation
        // offset ypeak and xpeak to fix
        if (ypeak + ysearch < ny_corr2/2)
            ypeak = ny_corr2 / 2 - ysearch;
        else if (ypeak + ysearch >= ny_corr - ny_corr2/2)
            ypeak = ny_corr - ny_corr2/2 - ysearch - 1;

        if (xpeak + xsearch < nx_corr2/2)
            xpeak = nx_corr2 / 2 - xsearch;
        else if (xpeak + xsearch >= nx_corr - nx_corr2/2)
            xpeak = nx_corr - nx_corr2/2 - xsearch - 1;

        corr2 = f64_array_slice(
                corr, nx_corr,
                ypeak + ysearch - ny_corr2/2, ny_corr2,
                xpeak + xsearch - nx_corr2/2, nx_corr2);

        for (int i=0; i<nx_corr2*ny_corr2; i++)
            corr2[i] = pow(corr2[i], 0.25);

        hi_corr = rdft_interpolate_2d(corr2, ny_corr2, nx_corr2, factor, factor, lock);

        int ny_hi = ny_corr2 * factor;
        int nx_hi = nx_corr2 * factor;
        int xpeak2, ypeak2;

        f64_array_stats(hi_corr, ny_hi, nx_hi, NULL, NULL, &ypeak2, &xpeak2);
        ypeak2 -= ny_hi / 2;
        xpeak2 -= nx_hi / 2;

        assert(xpeak2 >= -nx_hi/2 && xpeak2 < nx_hi/2);
        assert(ypeak2 >= -ny_hi/2 && ypeak2 < ny_hi/2);

        xfrac = xpeak2 / (double)factor;
        yfrac = ypeak2 / (double)factor;
    }

    data->xoff = data->xc->x_offset - ((xpeak + xfrac) / data->xc->ri);
    data->yoff = data->xc->y_offset - (ypeak + yfrac) + data->loc_y * data->xc->astretcha;
    data->corr = max_corr;

    // printf(" %d %6.3f %d %6.3f %6.2f \n", data->loc_x, xoff, data->loc_y, yoff, cmax);

    if (lock) pthread_mutex_lock(lock);
    fftw_free(c1);
    fftw_free(c2);
    fftw_free(c3);
    fftw_free(corr);
    if (lock) pthread_mutex_unlock(lock);
}

void do_correlation(struct st_xcorr *xc, long thread_n) {
    int loc_n, loc_x, loc_y;
    int slave_loc_x, slave_loc_y;
    int x_inc, y_inc;
    int nx_win, ny_win;
    int nx_corr, ny_corr;
    complex double *m_rows, *s_rows;
    complex double *c1, *c2;
    FILE *fmaster, *fslave;

    if ((fmaster = fopen(xc->m_path, "rb")) == NULL) {
        perror("failed to open master SLC image");
        exit(-1);
    }

    if ((fslave = fopen(xc->s_path, "rb")) == NULL) {
        perror("failed to open slave SLC image");
        exit(-1);
    }

    nx_corr = xc->xsearch * 2;
    nx_win = nx_corr * 2;
    ny_corr = xc->ysearch * 2;
    ny_win = ny_corr * 2;

    x_inc = (xc->m_nx - 2*(xc->xsearch + nx_corr)) / (xc->nxl + 3);
    y_inc = (xc->m_ny - 2*(xc->ysearch + ny_corr)) / (xc->nyl + 1);

    loc_n = loc_x = loc_y = 0;
    struct st_corr_thread_data thread_data[xc->nyl * xc->nxl];
    memset(thread_data, 0, sizeof(thread_data));

#ifndef NO_PTHREAD
    GThreadPool *thread_pool;
    pthread_mutex_t fftw_lock;

    thread_pool = g_thread_pool_new(corr_thread, &fftw_lock, thread_n, TRUE, NULL);
    pthread_mutex_init(&fftw_lock, NULL);
#endif

    for (int j=1; j<=xc->nyl; j++) {
        loc_y = ny_win + j * y_inc;
        slave_loc_y = (1+xc->astretcha)*loc_y + xc->y_offset;

        m_rows = load_slc_rows(fmaster, loc_y-ny_win/2, ny_win, xc->m_nx);
        s_rows = load_slc_rows(fslave, slave_loc_y-ny_win/2, ny_win, xc->s_nx);

        for (int i=2; i<=xc->nxl+1; i++) {
            loc_x = nx_win + i * x_inc;
            slave_loc_x = (1+xc->astretcha)*loc_x + xc->x_offset;

            //fprintf(stderr, "LOC#%d (%d, %d) <=> (%d, %d)\n", loc_n, loc_x, loc_y, slave_loc_x, slave_loc_y);

            c1 = c64_array_slice(m_rows, xc->m_nx, 0, ny_win, loc_x-nx_win/2, nx_win);
            c2 = c64_array_slice(s_rows, xc->s_nx, 0, ny_win, slave_loc_x-nx_win/2, nx_win);
 
            struct st_corr_thread_data *p = &thread_data[loc_n++];
            *p = (struct st_corr_thread_data) {
                .xc = xc,
                .c1 = c1,
                .c2 = c2,
                .loc_x = loc_x,
                .loc_y = loc_y
            };

#ifndef NO_PTHREAD
            g_thread_pool_push(thread_pool, p, NULL);
#else
            corr_thread(p, NULL);
#endif
        }

        fftw_free(m_rows);
        fftw_free(s_rows);
    }

#ifndef NO_PTHREAD
    g_thread_pool_free(thread_pool, FALSE, TRUE);
    pthread_mutex_destroy(&fftw_lock);
#endif
    fftw_cleanup();

    for (int i=0; i<loc_n; i++) {
        struct st_corr_thread_data *p = thread_data + i;
        printf(" %d %6.3f %d %6.3f %6.2f \n", p->loc_x, p->xoff, p->loc_y, p->yoff, p->corr);
    }
}

int main(int argc, char **argv) {
    struct st_xcorr xcorr;
    long thread_n;

    xcorr.m_path = argv[1];
    xcorr.s_path = argv[2];
    xcorr.m_nx = xcorr.s_nx = 11304;
    xcorr.m_ny = xcorr.s_ny = 27648;
    xcorr.xsearch = xcorr.ysearch = 64;
    xcorr.nxl = 16;
    //xcorr.nxl = 2;
    xcorr.nyl = 32;
    //xcorr.nyl = 4;
    xcorr.x_offset = -129;
    xcorr.y_offset = 62;
    xcorr.astretcha = 0;
    xcorr.ri = 2;
    xcorr.interp_flag = true;
    xcorr.interp_factor = 16;
    xcorr.n2x = xcorr.n2y = 8;

    thread_n = sysconf(_SC_NPROCESSORS_ONLN);
    fprintf(stderr, "use %ld thread(s)\n", thread_n);
    do_correlation(&xcorr, thread_n);

    //free(xcorr.master);
    //free(xcorr.slave);

    return 0;
}
