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

#include "xcorr2.h"
#include "xcorr2_args.h"

struct st_corr_thread_data {
    const struct st_xcorr *xc;
    complex double *c1;
    complex double *c2;
    double xoff, yoff;
    int loc_x, loc_y;
    double corr;
};

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

long double time_corr(
        const double *c1r,
        const double *c2r,
        int xsearch, int ysearch,
        int xoff, int yoff) {

    int nx_corr, ny_corr;
    int nx_win;

    nx_corr = xsearch * 2;
    nx_win = nx_corr * 2;
    ny_corr = ysearch * 2;

    long double num, denom, denom1, denom2, result;

    num = denom1 = denom2 = 0.0;
    for (int i=0; i<ny_corr; i++)
        for (int j=0; j<nx_corr; j++) {
            long double a = c1r[(ysearch + i + yoff) * nx_win + (xsearch + j + xoff)];
            long double b = c2r[(ysearch + i) * nx_win + (xsearch + j)];

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

double *freq_corr(
        double *c1r,
        double *c2r,
        int nx_win, int ny_win,
        pthread_mutex_t *fftw_lock) {
    complex double *c1r_fft, *c2r_fft;
    double *c3r;
    fftw_plan plan1, plan2, plan3;

    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    c1r_fft = fftw_alloc_complex(ny_win * (nx_win/2+1));
    c2r_fft = fftw_alloc_complex(ny_win * (nx_win/2+1));
    c3r = fftw_alloc_real(nx_win * ny_win);

    plan1 = fftw_plan_dft_r2c_2d(ny_win, nx_win, c1r, c1r_fft, FFTW_ESTIMATE);
    plan2 = fftw_plan_dft_r2c_2d(ny_win, nx_win, c2r, c2r_fft, FFTW_ESTIMATE);
    plan3 = fftw_plan_dft_c2r_2d(ny_win, nx_win, c1r_fft, c3r, FFTW_ESTIMATE);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    fftw_execute(plan1);
    fftw_execute(plan2);

    int isign = 1;
    for (int k=0; k<ny_win*(nx_win/2+1); k++, isign=-isign)
        c1r_fft[k] *= isign * conj(c2r_fft[k]);

    fftw_execute(plan3);

    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    fftw_free(c1r_fft);
    fftw_free(c2r_fft);
    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);
    fftw_destroy_plan(plan3);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    // FIXME: remove scaling later
    // scale to match GMTSAR for debugging
    for (int i=0; i<nx_win*ny_win; i++)
        c3r[i] = fabs(c3r[i] / (nx_win * ny_win));

    return c3r;
}

void corr_thread(gpointer arg, gpointer user_data) {
    struct st_corr_thread_data *data = arg;
    pthread_mutex_t *lock = user_data;

    int xsearch, ysearch;
    int nx_corr, ny_corr;
    int nx_win, ny_win;
    complex double *c1, *c2;
    double *c1r, *c2r;

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
        complex double *interp1, *interp2;
        int interp_width;

        interp1 = dft_interpolate_2d(c1, ny_win, nx_win, 1, data->xc->ri, lock);
        interp2 = dft_interpolate_2d(c2, ny_win, nx_win, 1, data->xc->ri, lock);
        interp_width = data->xc->ri * nx_win;

        if (lock) pthread_mutex_lock(lock);
        fftw_free(c1);
        fftw_free(c2);
        if (lock) pthread_mutex_unlock(lock);

        c1 = c64_array_slice(interp1, interp_width,
                0, ny_win, interp_width/2 - nx_win/2, nx_win);
        c2 = c64_array_slice(interp2, interp_width,
                0, ny_win, interp_width/2 - nx_win/2, nx_win);

        if (lock) pthread_mutex_lock(lock);
        fftw_free(interp1);
        fftw_free(interp2);
        if (lock) pthread_mutex_unlock(lock);
    }

    c1r = fftw_alloc_real(nx_win * ny_win);
    c2r = fftw_alloc_real(nx_win * ny_win);

    double mean1 = 0.0, mean2 = 0.0;
    for (int k=0; k<nx_win*ny_win; k++) {
        mean1 += (c1r[k] = cabs(c1[k]));
        mean2 += (c2r[k] = cabs(c2[k]));
    }

    if (lock) pthread_mutex_lock(lock);
    fftw_free(c1);
    fftw_free(c2);
    if (lock) pthread_mutex_unlock(lock);

    mean1 /= nx_win * ny_win;
    mean2 /= nx_win * ny_win;
    for (int k=0; k<nx_win*ny_win; k++) {
        c1r[k] -= mean1;
        c2r[k] -= mean2;
    }

    // make_mask and mask
    for (int i=0; i<ny_win; i++)
        for (int j=0; j<nx_win; j++) {
            if (i < ysearch
                    || i >= ny_win - ysearch
                    || j < xsearch
                    || j >= nx_win - xsearch)
                c2r[i*nx_win + j] = 0;
        }

    // calc correlation with 2D FFT
    double *c3r, *corr;
    c3r = freq_corr(c1r, c2r, nx_win, ny_win, lock);
    corr = f64_array_slice(c3r, nx_win, ysearch, ny_corr, xsearch, nx_corr);

    //puts("ARRAY corr:");
    //print_complex_double("%+04.2f%+04.2fj\t", corr, ny_corr, nx_corr);

    int xpeak, ypeak;
    double cmax, cave, max_corr;

    f64_array_stats(corr, ny_corr, nx_corr, &cave, &cmax, &ypeak, &xpeak);
    xpeak -= xsearch;
    ypeak -= ysearch;

    max_corr = time_corr(c1r, c2r, xsearch, ysearch, xpeak, ypeak);

    if (lock) pthread_mutex_lock(lock);
    fftw_free(c1r);
    fftw_free(c2r);
    if (lock) pthread_mutex_unlock(lock);

    //fprintf(stderr, "xypeak: (%d, %d)\n", xpeak, ypeak);
    //fprintf(stderr, "max_corr: %g\n", cmax);

    double xfrac = 0.0, yfrac = 0.0;

    // high-res correlation
    if (data->xc->interp_factor > 1) {
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

        if (lock) pthread_mutex_lock(lock);
        fftw_free(corr2);
        fftw_free(hi_corr);
        if (lock) pthread_mutex_unlock(lock);
    }

    data->xoff = data->xc->x_offset - ((xpeak + xfrac) / data->xc->ri);
    data->yoff = data->xc->y_offset - (ypeak + yfrac) + data->loc_y * data->xc->astretcha;
    data->corr = max_corr;

    // printf(" %d %6.3f %d %6.3f %6.2f \n", data->loc_x, xoff, data->loc_y, yoff, cmax);

    if (lock) pthread_mutex_lock(lock);
    fftw_free(c3r);
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

    FILE *fout = fopen("freq_xcorr.dat", "w");
    for (int i=0; i<loc_n; i++) {
        struct st_corr_thread_data *p = thread_data + i;
        fprintf(fout, " %d %6.3f %d %6.3f %6.2f \n",
                p->loc_x, p->xoff, p->loc_y, p->yoff, p->corr);
    }
    fclose(fout);
}

int main(int argc, char **argv) {
    struct st_xcorr_args args;
    struct st_xcorr xcorr;
    long thread_n = 0;

    parse_opts(&args, argc, argv);
    apply_args(&args, &xcorr);

#ifndef NO_PTHREAD
    thread_n = sysconf(_SC_NPROCESSORS_ONLN);
    fprintf(stderr, "use %ld thread(s)\n", thread_n);
#endif
    do_correlation(&xcorr, thread_n);

    free(xcorr.m_path);
    free(xcorr.s_path);

    return 0;
}
