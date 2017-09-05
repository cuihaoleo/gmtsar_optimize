#include <complex.h>

extern "C" {
#include "xcorr2.h"
#include "xcorr2_args.h"
}

#undef complex
#include <arrayfire.h>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <complex>
#include <cmath>

af::array load_slc_rows(std::ifstream &fin, int start, int n_rows, int nx) {
    long offset;

    offset = nx * start * sizeof(short) * 2;
    fin.seekg(offset, fin.beg);

    int16_t *buf = new int16_t[n_rows * nx * 2];
    fin.read((char*)buf, n_rows * nx * sizeof(int16_t) * 2);

    af::array af_buf(2, nx, n_rows, buf);
    af::array dest = af::complex(af_buf(0, af::span, af::span), af_buf(1, af::span, af::span));
    dest = af::moddims(dest, nx, n_rows);
    delete[] buf;

    return af::transpose(dest);
}

af::array dft_interpolate(const af::array &in, int scale_h, int scale_w) {
    int height = in.dims(0);
    int width = in.dims(1);
    int out_height = height * scale_h;
    int out_width = width * scale_w;

    af::array in_fft = af::dft(in);
    af::array out_fft = af::constant(af::cdouble(0, 0), out_height, out_width, c32);

    af::seq left = af::seq(0, width/2);
    af::seq &out_left = left;
    af::seq right = af::seq(width/2, width-1);
    af::seq out_right = af::seq(out_width-width/2, out_width-1);;
    af::seq up = af::seq(0, height/2-1);
    af::seq &out_up = up;
    af::seq down = af::seq(height/2, height-1);
    af::seq out_down = af::seq(out_height-height/2, out_height-1);;

    in_fft(af::span, width/2) /= 2.0;
    out_fft(out_up, out_left) = in_fft(up, left);
    out_fft(out_up, out_right) = in_fft(up, right);
    out_fft(out_down, out_left) = in_fft(down, left);
    out_fft(out_down, out_right) = in_fft(down, right);

    return af::idft(out_fft, 1.0/(height * width), out_fft.dims());
}

int main(int argc, char **argv) {
    st_xcorr_args args;
    st_xcorr xcorr;
    parse_opts(&args, argc, argv);
    apply_args(&args, &xcorr);

    std::ifstream f1(xcorr.m_path, std::ios::binary);
    std::ifstream f2(xcorr.s_path, std::ios::binary);

    const int xsearch = xcorr.xsearch;
    const int ysearch = xcorr.ysearch;
    const int nx_corr = xcorr.xsearch * 2;
    const int nx_win = nx_corr * 2;
    const int ny_corr = xcorr.ysearch * 2;
    const int ny_win = ny_corr * 2;
    const int x_inc = (xcorr.m_nx - 2*(xcorr.xsearch + nx_corr)) / (xcorr.nxl + 3);
    const int y_inc = (xcorr.m_ny - 2*(xcorr.ysearch + ny_corr)) / (xcorr.nyl + 1);

    int loc_x, loc_y;
    int slave_loc_x, slave_loc_y;
    loc_x = loc_y = slave_loc_x = slave_loc_y = 0;

    switch (args.device) {
        case XCORR2_DEVICE_CUDA:
            af::setBackend(AF_BACKEND_CUDA);
            break;
        case XCORR2_DEVICE_OPENCL:
            af::setBackend(AF_BACKEND_OPENCL);
            break;
        case XCORR2_DEVICE_CPU:
            af::setBackend(AF_BACKEND_CPU);
            break;
        default:
            af::setBackend(AF_BACKEND_DEFAULT);
    }
    //af::info();

    int *corr_mask_arr = new int[nx_win * ny_win];
    for (int i=0; i<nx_win; i++)
        for (int j=0; j<ny_win; j++)
            corr_mask_arr[i*ny_win + j] = ((i + j) & 1) ? -1 : 1;

    af::array corr_mask(ny_win, nx_win, corr_mask_arr);
    delete[] corr_mask_arr;

    af::array m_rows;
    af::array s_rows;
    for (int j=1; j<=xcorr.nyl; j++) {
        loc_y = ny_win + j * y_inc;
        slave_loc_y = (1+xcorr.astretcha)*loc_y + xcorr.y_offset;

        m_rows = load_slc_rows(f1, loc_y-ny_win/2, ny_win, xcorr.m_nx);
        s_rows = load_slc_rows(f2, slave_loc_y-ny_win/2, ny_win, xcorr.m_nx);

        for (int i=2; i<=xcorr.nxl+1; i++) {
            loc_x = nx_win + i * x_inc;
            slave_loc_x = (1+xcorr.astretcha)*loc_x + xcorr.x_offset;

            const af::seq slice_x(loc_x - nx_win/2, loc_x + nx_win/2 - 1);
            const af::seq slave_slice_x(slave_loc_x - nx_win/2, slave_loc_x + nx_win/2 - 1);

            af::array c1, c2;
            c1 = m_rows(af::span, slice_x);
            c2 = s_rows(af::span, slave_slice_x);

            if (xcorr.ri > 1) {
                af::array interp1, interp2;
                int interp_width = xcorr.ri * nx_win;

                interp1 = dft_interpolate(c1, 1, xcorr.ri);
                interp2 = dft_interpolate(c2, 1, xcorr.ri);

                const af::seq x_seq(interp_width/2 - nx_win/2, interp_width/2 + nx_win/2 - 1);
                c1 = interp1(af::span, x_seq);
                c2 = interp2(af::span, x_seq);
            }

            af::array c1r = af::abs(c1);
            af::array c2ro = af::abs(c2);

            float m1 = af::mean<float>(c1r);
            float m2 = af::mean<float>(c2ro);

            c1r -= m1;
            c2ro -= m2;

            af::array c2r = af::constant(0, ny_win, nx_win, f32);
            af::seq roi_y(ysearch, ny_win - ysearch - 1);
            af::seq roi_x(xsearch, nx_win - xsearch - 1);
            c2r(roi_y, roi_x) = c2ro(roi_y, roi_x);

            af::array c1r_fft = af::dft(c1r);
            af::array c2r_fft = af::dft(c2r);
            af::array c3r_fft = c1r_fft * corr_mask * af::conjg(c2r_fft);
            af::array c3r = af::idft(c3r_fft, 1.0/(nx_win*ny_win), c3r_fft.dims());
            af::array corr = c3r(
                    af::seq(ysearch, ysearch + ny_corr - 1),
                    af::seq(xsearch, xsearch + nx_corr - 1));
            corr = af::abs(corr);

            unsigned max_idx;
            float cmax;
            af::max<float>(&cmax, &max_idx, corr);

            int xpeak = max_idx / ny_corr - xsearch;
            int ypeak = max_idx % ny_corr - ysearch;
            af::array core1 = c1r(
                af::seq(ysearch + ypeak, ysearch + ypeak + ny_corr - 1),
                af::seq(xsearch + xpeak, xsearch + xpeak + nx_corr - 1));
            af::array core2 = c2r(
                af::seq(ysearch, ysearch + ny_corr - 1),
                af::seq(xsearch, xsearch + nx_corr - 1));
            float denom1 = af::norm(core1);
            float denom2 = af::norm(core2);
            float num = af::sum<float>(core1 * core2);
            float max_corr = 100 * fabs(num / (denom1 * denom2));
            // float max_corr = fabs(af::corrcoef<float>(core1, core2)) * 100;

            float xfrac = 0.0, yfrac = 0.0;
            if (xcorr.interp_factor > 1) {
                int factor = xcorr.interp_factor;
                int nx_corr2 = xcorr.n2x;
                int ny_corr2 = xcorr.n2y;

                // FIXME: remove this later
                // scale to match GMTSAR for debugging
                corr *= max_corr / cmax;

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

                af::array corr2 = corr(
                        af::seq(ypeak + ysearch - ny_corr2/2, ypeak + ysearch + ny_corr2/2 - 1),
                        af::seq(xpeak + xsearch - nx_corr2/2, xpeak + xsearch + nx_corr2/2 - 1));
                corr2 = af::pow(corr2, 0.25);

                af::array hi_corr = dft_interpolate(corr2, factor, factor);
                hi_corr = af::abs(hi_corr);
         
                int ny_hi = ny_corr2 * factor;
                int nx_hi = nx_corr2 * factor;

                unsigned max_idx;
                float cmax;
                af::max<float>(&cmax, &max_idx, hi_corr);

                int xpeak2 = max_idx / ny_hi - nx_hi / 2;
                int ypeak2 = max_idx % ny_hi - ny_hi / 2;

                assert(xpeak2 >= -nx_hi/2 && xpeak2 < nx_hi/2);
                assert(ypeak2 >= -ny_hi/2 && ypeak2 < ny_hi/2);

                xfrac = xpeak2 / (float)factor;
                yfrac = ypeak2 / (float)factor;
            }

            float xoff = xcorr.x_offset - ((xpeak + xfrac) / xcorr.ri);
            float yoff = xcorr.y_offset - (ypeak + yfrac) + loc_y * xcorr.astretcha;
            printf(" %d %6.3lf %d %6.3lf %6.2lf \n", loc_x, xoff, loc_y, yoff, max_corr);
        }
    }

    f1.close();
    f2.close();

    return 0;
}
