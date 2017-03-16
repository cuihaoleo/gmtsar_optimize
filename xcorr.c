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
    complex float *c1;
    complex float *c2;
    float xoff, yoff;
    int loc_x, loc_y;
    float corr;
};

void print_complex_float_array(const char *fmt, complex float *arr, int n, int m) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            printf(fmt, crealf(arr[i*m+j]), cimagf(arr[i*m+j]));
        }
        putchar('\n');
    }
}

complex float *load_slc_rows(FILE *fin, int start, int n_rows, int nx) {
    long offset;
    short *tmp;
    complex float *arr;

    offset = nx * start * sizeof(short) * 2;
    fseek(fin, offset, SEEK_SET);

    tmp = malloc(nx * sizeof(short) * 2);
    arr = fftwf_malloc(n_rows * nx * sizeof(complex float));

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

complex float *c32_array_slice(
        const complex float *mat, int n_cols,
        int tl_y, int s_y,
        int tl_x, int s_x) {

    assert(tl_y >= 0 && tl_x >= 0 && tl_x < n_cols);

    complex float *arr;
    arr = fftwf_malloc(s_x * s_y * sizeof(complex float));

    for (int i=0; i<s_y; i++)
        memcpy(&arr[i*s_x], &mat[(i+tl_y)*n_cols + tl_x], s_x * sizeof(complex float));

    return arr;
}

complex float *dft_interpolate(
        complex float *in,
        int length,
        int scale,
        pthread_mutex_t *fftw_lock) {

    fftwf_plan plan1, plan2;
    complex float *in_fft;
    complex float *out;

    assert(length >= 2 && TEST_2PWR(length));
    assert(scale >= 2 && TEST_2PWR(scale));
    
    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    in_fft = fftwf_malloc(length * sizeof(complex float) * scale);
    out = fftwf_malloc(length * sizeof(complex float) * scale);
    plan1 = fftwf_plan_dft_1d(length, in, in_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    plan2 = fftwf_plan_dft_1d(length * scale, in_fft, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    fftwf_execute(plan1);

    int tail_offset = (scale - 1) * length;
    memcpy(&in_fft[length/2 + tail_offset], &in_fft[length/2], length/2 * sizeof(complex float));
    memset(&in_fft[length/2], 0, tail_offset * sizeof(complex float));

    fftwf_execute(plan2);

    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    fftwf_free(in_fft);
    fftwf_destroy_plan(plan1);
    fftwf_destroy_plan(plan2);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    for (int i=0; i<scale*length; i++)
        out[i] /= length;

    return out;
}

complex float *dft_interpolate_2d(
        complex float *in,
        int height, int width,
        int scale_h, int scale_w,
        pthread_mutex_t *fftw_lock) {

    fftwf_plan plan1, plan2;
    int out_height, out_width;
    complex float *in_fft;
    complex float *out_fft;
    complex float *out;

    assert(height >= 2 && TEST_2PWR(height));
    assert(scale_h >= 2 && TEST_2PWR(scale_h));
    assert(width >= 2 && TEST_2PWR(width));
    assert(scale_w >= 2 && TEST_2PWR(scale_w));

    out_height = height * scale_h;
    out_width = width * scale_w;
    
    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    in_fft = fftwf_malloc(height * width * sizeof(complex float));
    out_fft = fftwf_malloc(out_height * out_width * sizeof(complex float));
    out = fftwf_malloc(out_height * out_width * sizeof(complex float));
    plan1 = fftwf_plan_dft_2d(height, width, in, in_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    plan2 = fftwf_plan_dft_2d(out_height, out_width, out_fft, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    fftwf_execute(plan1);

    int y_offset = out_height - height;
    int x_offset = out_width - width;
    memset(out_fft, 0, out_height * out_width * sizeof(complex float));

    for (int y=0; y<out_width; y++)
        for (int x=0; x<out_height; x++) {
            int sx, sy;

            if (x < width / 2)
                sx = x;
            else if (x >= out_width - width/2)
                sx = x - x_offset;
            else
                continue;

            if (y < height / 2)
                sy = y;
            else if (y >= out_height - height/2)
                sy = y - y_offset;
            else
                continue;

            assert(sx >= 0 && sx < width);
            assert(sy >= 0 && sy < height);

            out_fft[y*out_height + x] = in_fft[sy*height + sx];
        }

    fftwf_execute(plan2);

    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    fftwf_free(in_fft);
    fftwf_free(out_fft);
    fftwf_destroy_plan(plan1);
    fftwf_destroy_plan(plan2);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    // FIXME: remove this later
    // scale to match GMTSAR for debugging
    for (int i=0; i<out_height*out_width; i++)
        out[i] /= height * width;

    return out;
}

long double time_corr(
        complex float *c1,
        complex float *c2,
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

complex float *freq_corr(
        complex float *c1,
        complex float *c2,
        int nx_win, int ny_win,
        pthread_mutex_t *fftw_lock) {
    complex float *c1_fft, *c2_fft, *c3;
    fftwf_plan plan1, plan2, plan3;

    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    c1_fft = fftwf_malloc(nx_win * ny_win * sizeof(complex float));
    c2_fft = fftwf_malloc(nx_win * ny_win * sizeof(complex float));
    c3 = fftwf_malloc(nx_win * ny_win * sizeof(complex float));

    plan1 = fftwf_plan_dft_2d(ny_win, nx_win, c1, c1_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    plan2 = fftwf_plan_dft_2d(ny_win, nx_win, c2, c2_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    plan3 = fftwf_plan_dft_2d(ny_win, nx_win, c1_fft, c3, FFTW_BACKWARD, FFTW_ESTIMATE);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    fftwf_execute(plan1);
    fftwf_execute(plan2);

    int isign = 1;
    for (int i=0; i<ny_win; i++) {
        for (int j=0; j<nx_win; j++) {
            c1_fft[i*nx_win + j] *= isign * conjf(c2_fft[i*nx_win + j]);
            isign = -isign;
        }

        isign = -isign;
    }

    fftwf_execute(plan3);

    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    fftwf_free(c1_fft);
    fftwf_free(c2_fft);
    fftwf_destroy_plan(plan1);
    fftwf_destroy_plan(plan2);
    fftwf_destroy_plan(plan3);
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

    float mean1, mean2;
    int xsearch, ysearch;
    int nx_corr, ny_corr;
    int nx_win, ny_win;
    complex float *c1, *c2;

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
            complex float *interp1 = dft_interpolate(c1 + i*nx_win, nx_win, data->xc->ri, lock);
            complex float *interp2 = dft_interpolate(c2 + i*nx_win, nx_win, data->xc->ri, lock);
            int offset = data->xc->ri * nx_win / 2 - nx_win / 2;

            memcpy(&c1[i*nx_win], interp1 + offset, nx_win * sizeof(complex float));
            memcpy(&c2[i*nx_win], interp2 + offset, nx_win * sizeof(complex float));

            if (lock) pthread_mutex_lock(lock);
            fftwf_free(interp1);
            fftwf_free(interp2);
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
    complex float *c3;
    c3 = freq_corr(c1, c2, nx_win, ny_win, lock);

    complex float *corr;
    corr = c32_array_slice(c3, nx_win, ysearch, ny_corr, xsearch, nx_corr);

    //puts("ARRAY corr:");
    //print_complex_float_array("%+04.2f%+04.2fj\t", corr, ny_corr, nx_corr);

    int xpeak, ypeak;
    double cmax, cave, max_corr;

    cave = 0;
    xpeak = ypeak = INT_MIN;
    cmax = -INFINITY;

    for (int i=0, k=0; i<ny_corr; i++)
        for (int j=0; j<nx_corr; j++) {
            double norm = cabs(corr[k++]);
            cave += norm;

            if (norm > cmax) {
                cmax = norm;
                xpeak = j - xsearch;
                ypeak = i - ysearch;
            }
        }

    assert(xpeak >= INT_MIN && ypeak >= INT_MIN);

    cave /= nx_corr * ny_corr;
    max_corr = time_corr(c1, c2, xsearch, ysearch, xpeak, ypeak);

    //fprintf(stderr, "xypeak: (%d, %d)\n", xpeak, ypeak);
    //fprintf(stderr, "max_corr: %g\n", cmax);

    float xfrac = 0.0, yfrac = 0.0;

    // high-res correlation
    if (data->xc->interp_flag) {
        int factor = data->xc->interp_factor;
        int nx_corr2 = data->xc->n2x;
        int ny_corr2 = data->xc->n2y;
        complex float *corr2;
        complex float *hi_corr;

        assert(nx_corr2 >= 2 && TEST_2PWR(nx_corr2));
        assert(ny_corr2 >= 2 && TEST_2PWR(ny_corr2));

        // FIXME: remove this later
        // scale to match GMTSAR for debugging
        for (int i=0; i<nx_corr*ny_corr; i++)
            corr[i] = cabs(corr[i]) * max_corr / cmax;

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

        corr2 = c32_array_slice(
                corr, nx_corr,
                ypeak + ysearch - ny_corr2/2, ny_corr2,
                xpeak + xsearch - nx_corr2/2, nx_corr2);

        for (int i=0; i<nx_corr2*ny_corr2; i++)
            corr2[i] = powf(corr2[i], 0.25);

        hi_corr = dft_interpolate_2d(corr2, ny_corr2, nx_corr2, factor, factor, lock);

        int xpeak2, ypeak2;
        int ny_hi, nx_hi;
        double cmax;

        xpeak2 = ypeak2 = INT_MIN;
        ny_hi = ny_corr2 * factor;
        nx_hi = nx_corr2 * factor;
        cmax = -INFINITY;

        for (int i=0, k=0; i<ny_hi; i++)
            for (int j=0; j<nx_hi; j++) {
                // FIXME: why not cabs? GMTSAR do this
                double norm = hi_corr[k++];

                if (norm > cmax) {
                    cmax = norm;
                    xpeak2 = j - nx_hi/2;
                    ypeak2 = i - ny_hi/2;
                }
            }

        assert(xpeak2 >= -nx_hi/2 && xpeak2 < nx_hi/2);
        assert(ypeak2 >= -ny_hi/2 && ypeak2 < ny_hi/2);
        xfrac = xpeak2 / (float)factor;
        yfrac = ypeak2 / (float)factor;
    }

    data->xoff = data->xc->x_offset - ((xpeak + xfrac) / data->xc->ri);
    data->yoff = data->xc->y_offset - (ypeak + yfrac) + data->loc_y * data->xc->astretcha;
    data->corr = max_corr;

    // printf(" %d %6.3f %d %6.3f %6.2f \n", data->loc_x, xoff, data->loc_y, yoff, cmax);

    if (lock) pthread_mutex_lock(lock);
    fftwf_free(c1);
    fftwf_free(c2);
    fftwf_free(c3);
    fftwf_free(corr);
    if (lock) pthread_mutex_unlock(lock);
}


void do_correlation(struct st_xcorr *xc, long thread_n) {
    int loc_n, loc_x, loc_y;
    int slave_loc_x, slave_loc_y;
    int x_inc, y_inc;
    int nx_win, ny_win;
    int nx_corr, ny_corr;
    complex float *m_rows, *s_rows;
    complex float *c1, *c2;
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

            c1 = c32_array_slice(m_rows, xc->m_nx, 0, ny_win, loc_x-nx_win/2, nx_win);
            c2 = c32_array_slice(s_rows, xc->s_nx, 0, ny_win, slave_loc_x-nx_win/2, nx_win);
 
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

        fftwf_free(m_rows);
        fftwf_free(s_rows);
    }

#ifndef NO_PTHREAD
    g_thread_pool_free(thread_pool, FALSE, TRUE);
    pthread_mutex_destroy(&fftw_lock);
#endif
    fftwf_cleanup();

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
    xcorr.n2x = 8;
    xcorr.n2y = 8;

    //xcorr.master = load_SLC_c32(master_file, 11304, 27648);
    //xcorr.slave = load_SLC_c32(slave_file, 11304, 27648);

    thread_n = sysconf(_SC_NPROCESSORS_ONLN);
    fprintf(stderr, "use %ld thread(s)\n", thread_n);
    do_correlation(&xcorr, thread_n);

    //free(xcorr.master);
    //free(xcorr.slave);

    return 0;
}
