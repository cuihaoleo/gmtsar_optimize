#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <complex.h>
#include <pthread.h>
#include <fftw3.h>

#define TEST_2PWR(n) ((n) > 0 && ((n) & ((n) - 1)) == 0)

struct st_xcorr {
    int m_nx, m_ny;
    int s_nx, s_ny;

    int x_offset, y_offset;
    int xsearch, ysearch;
    int nxl, nyl;
    int astretcha;

    int ri;

    char *m_path;
    char *s_path;
};

struct st_corr_thread_data {
    const struct st_xcorr *xc;
    complex float *c1;
    complex float *c2;
    int xoff, yoff;
    int loc_x, loc_y;
    float corr;
};

void print_complex_float_array(const char *fmt, complex float *arr, int n, int m) {
    int i, j;

    for (i=0; i<n; i++) {
        for (j=0; j<m; j++) {
            printf(fmt, crealf(arr[i*m+j]), cimagf(arr[i*m+j]));
        }
        putchar('\n');
    }
}

complex float *load_slc_rows(FILE *fin, int start, int n_rows, int nx) {
    int i, j;
    long offset;
    short *tmp;
    complex float *arr;

    offset = nx * start * sizeof(short) * 2;
    fseek(fin, offset, SEEK_SET);

    tmp = malloc(nx * sizeof(short) * 2);
    arr = fftwf_malloc(n_rows * nx * sizeof(complex float));

    for (i=0; i<n_rows; i++) {
        if (fread(tmp, 2*sizeof(short), nx, fin) != (unsigned long)nx) {
            perror("Failed to read data from SLC file!");
            exit(-1);
        }

        for (j=0; j<nx; j++)
            arr[i*nx + j] = tmp[2*j] + tmp[2*j+1] * I;
    }

    free(tmp);
    return arr;
}

complex float *c32_array_slice(
        const complex float *mat, int n_cols,
        int tl_y, int s_y,
        int tl_x, int s_x) {
    complex float *arr;
    arr = fftwf_malloc(s_x * s_y * sizeof(complex float));

    for (int i=0; i<s_y; i++)
        for (int j=0; j<s_x; j++)
            arr[i*s_x + j] = mat[(i+tl_y)*n_cols + j + tl_x];

    return arr;
}

complex float *dft_interpolate(
        complex float *in,
        int length,
        int scale) {

    fftwf_plan plan;
    complex float *in_fft;
    complex float *out;

    assert(length >= 2 && TEST_2PWR(length));
    assert(scale >= 2 && TEST_2PWR(scale));
    
    in_fft = fftwf_malloc(length * sizeof(complex float) * scale);
    out = fftwf_malloc(length * sizeof(complex float) * scale);

    plan = fftwf_plan_dft_1d(length, in, in_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);

    int tail_offset = (scale - 1) * length;
    for (int i=length/2; i<length; i++)
        in_fft[i + tail_offset] = in_fft[i];

    memset(&in_fft[length/2], 0, tail_offset * sizeof(complex float));

    fftwf_destroy_plan(plan);
    plan = fftwf_plan_dft_1d(length * scale, in_fft, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_free(in_fft);

    for (int i=0; i<scale*length; i++)
        out[i] /= length;

    return out;
}

long double time_corr(
        const struct st_xcorr *xc,
        complex float *c1,
        complex float *c2,
        int xoff, int yoff) {

    int xsearch, ysearch;
    int nx_corr, ny_corr;
    int nx_win, ny_win;

    xsearch = xc->xsearch;
    nx_corr = xsearch * 2;
    nx_win = nx_corr * 2;
    ysearch = xc->ysearch;
    ny_corr = ysearch * 2;
    ny_win = ny_corr * 2;

    long double num, denom, denom1, denom2, result;

    num = denom1 = denom2 = 0.0;
    for (int i=0; i<ny_corr; i++)
        for (int j=0; j<nx_corr; j++) {
            long double a = c1[(xc->ysearch + i + yoff) * nx_win + (xc->xsearch + j + xoff)];
            long double b = c2[(xc->ysearch + i) * nx_win + (xc->xsearch + j)];

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

void *corr_thread(void *arg) {
    struct st_corr_thread_data *data = arg;
    float mean1, mean2;
    int xsearch, ysearch;
    int nx_corr, ny_corr;
    int nx_win, ny_win;
    complex float *c1, *c2;
    int i, j, k;

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
            complex float *interp1 = dft_interpolate(c1 + i*nx_win, nx_win, data->xc->ri);
            complex float *interp2 = dft_interpolate(c2 + i*nx_win, nx_win, data->xc->ri);
            int offset = data->xc->ri * nx_win / 2 - nx_win / 2;

            for (int j=0; j<nx_win; j++) {
                c1[i*nx_win + j] = interp1[j+offset];
                c2[i*nx_win + j] = interp2[j+offset];
            }
        }
    }

    mean1 = mean2 = 0.0;
    for (i=0; i<nx_win*ny_win; i++) {
        c1[i] = cabs(c1[i]);
        c2[i] = cabs(c2[i]);

        mean1 += creal(c1[i]);
        mean2 += creal(c2[i]);
    }

    mean1 /= nx_win * ny_win;
    mean2 /= nx_win * ny_win;

    for (i=0; i<nx_win*ny_win; i++) {
        c1[i] -= mean1;
        c2[i] -= mean2;
    }

    // make_mask and mask
    for (i=0; i<ny_win; i++)
        for (j=0; j<nx_win; j++) {
            if (i < ysearch
                    || i >= ny_win - ysearch
                    || j < xsearch
                    || j >= nx_win - xsearch)
                c2[i*nx_win + j] = 0;
        }


    complex float *c1_fft = fftwf_malloc(nx_win * ny_win * sizeof(complex float));
    fftwf_plan plan1;
    complex float *c2_fft = fftwf_malloc(nx_win * ny_win * sizeof(complex float));
    fftwf_plan plan2;

    plan1 = fftwf_plan_dft_2d(ny_win, nx_win, c1, c1_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    plan2 = fftwf_plan_dft_2d(ny_win, nx_win, c2, c2_fft, FFTW_FORWARD, FFTW_ESTIMATE);

    fftwf_execute(plan1);
    fftwf_execute(plan2);

    int isign = 1;
    for (i=0; i<ny_win; i++) {
        for (j=0; j<nx_win; j++) {
            c1_fft[i*nx_win + j] *= isign * conjf(c2_fft[i*nx_win + j]);
            isign = -isign;
        }

        isign = -isign;
    }


    fftwf_free(c2_fft);
    fftwf_destroy_plan(plan1);
    fftwf_destroy_plan(plan2);

    //puts("ARRAY c3_fft:");
    //print_complex_float_array("%+04.2f%+04.2fj\t", c1_fft, ny_win, nx_win);

    complex float *c3 = fftwf_malloc(nx_win * ny_win * sizeof(complex float));
    plan1 = fftwf_plan_dft_2d(ny_win, nx_win, c1_fft, c3, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(plan1);
    fftwf_free(c1_fft);
    fftwf_destroy_plan(plan1);

    // TODO: scale IFFT result to match GMTSAR
    // just for debug, remove in the future
    for (i=0; i<nx_win*ny_win; i++)
        c3[i] /= nx_win * ny_win;

    //puts("ARRAY c3:");
    //print_complex_float_array("%+04.2f%+04.2fj\t", c3, ny_win, nx_win);

    complex float *corr;
    corr = c32_array_slice(c3, nx_win, ysearch, ny_corr, xsearch, nx_corr);
    //puts("ARRAY corr:");
    //print_complex_float_array("%+04.2f%+04.2fj\t", corr, ny_corr, nx_corr);

    int xpeak, ypeak;
    double cmax, cave;

    cave = k = 0;
    xpeak = ypeak = INT_MIN;
    cmax = -INFINITY;

    for (i=0; i<ny_corr; i++)
        for (j=0; j<nx_corr; j++) {
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
    cmax /= cave;

    cmax = time_corr(data->xc, c1, c2, xpeak, ypeak);

    fprintf(stderr, "xypeak: (%d, %d)\n", xpeak, ypeak);
    fprintf(stderr, "max_corr: %g\n", cmax);

    float xoff, yoff;
    float xfrac, yfrac;

    // TODO: sub-pixel high-res correlation
    xfrac = yfrac = 0.0;

    xoff = data->xc->x_offset - ((xpeak + xfrac) / data->xc->ri);
    yoff = data->xc->y_offset - (ypeak + yfrac) + data->loc_y * data->xc->astretcha;

    data->xoff = xoff;
    data->yoff = yoff;
    data->corr = cmax;


    printf(" %d %6.3f %d %6.3f %6.2f \n", data->loc_x, xoff, data->loc_y, yoff, cmax);

    fftwf_free(c1);
    fftwf_free(c2);
    fftwf_free(c3);

    return NULL;
}


void do_correlation(struct st_xcorr *xc) {
    int i, j;
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

#define NUM_THREADS 4
    //pthread_t threads[NUM_THREADS];
    //int thread_id = 0;

    for (j=1; j<=xc->nyl; j++) {
        loc_y = ny_win + j * y_inc;
        slave_loc_y = (1+xc->astretcha)*loc_y + xc->y_offset;

        m_rows = load_slc_rows(fmaster, loc_y-ny_win/2, ny_win, xc->m_nx);
        s_rows = load_slc_rows(fslave, slave_loc_y-ny_win/2, ny_win, xc->s_nx);

        for (i=2; i<=xc->nxl+1; i++) {
            loc_x = nx_win + i * x_inc;
            slave_loc_x = (1+xc->astretcha)*loc_x + xc->x_offset;

            fprintf(stderr, "LOC#%d (%d, %d) <=> (%d, %d)\n", loc_n, loc_x, loc_y, slave_loc_x, slave_loc_y);

            complex float *load_cols(complex float *rows, int n_rows, int nx, int start, int n_cols);
            //c1 = load_cols(m_rows, ny_win, xc->m_nx, loc_x-nx_win/2, nx_win);
            c1 = c32_array_slice(m_rows, xc->m_nx, 0, ny_win, loc_x-nx_win/2, nx_win);
            //c2 = load_cols(s_rows, ny_win, xc->s_nx, slave_loc_x-nx_win/2, nx_win);
            c2 = c32_array_slice(s_rows, xc->s_nx, 0, ny_win, slave_loc_x-nx_win/2, nx_win);
 
            struct st_corr_thread_data *p = &thread_data[loc_n++];
            p->xc = xc;
            p->c1 = c1;
            p->c2 = c2;
            p->loc_x = loc_x;
            p->loc_y = loc_y;

            corr_thread(p);
            /*int rc;
            rc = pthread_create(&threads[thread_id++], NULL, corr_thread, p);
            if (rc) {
                fprintf(stderr, "ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }*/

            //exit(-1);
        }
    }
}

int main(int argc, char **argv) {
    struct st_xcorr xcorr;
    xcorr.m_path = argv[1];
    xcorr.s_path = argv[2];
    xcorr.m_nx = xcorr.s_nx = 11304;
    xcorr.m_ny = xcorr.s_ny = 27648;
    xcorr.xsearch = xcorr.ysearch = 64;
    xcorr.nxl = 16;
    xcorr.nyl = 32;
    xcorr.x_offset = -129;
    xcorr.y_offset = 62;
    xcorr.astretcha = 0;
    xcorr.ri = 2;

    //xcorr.master = load_SLC_c32(master_file, 11304, 27648);
    //xcorr.slave = load_SLC_c32(slave_file, 11304, 27648);

    do_correlation(&xcorr);

    //free(xcorr.master);
    //free(xcorr.slave);

    return 0;
}
