#ifndef __XCORR2_ARGS_H_INCLUDED__
#define __XCORR2_ARGS_H_INCLUDED__

struct st_xcorr {
    int m_nx, m_ny;
    int s_nx, s_ny;

    int x_offset, y_offset;
    int xsearch, ysearch;
    int nxl, nyl;
    double astretcha;

    int ri;
    int interp_factor;
    int n2x, n2y;  // high-res correlation window

    char *m_path;
    char *s_path;
};

struct st_xcorr_args {
    const char *m_prm;
    const char *s_prm;

    int nx, ny;
    int xsearch, ysearch;
    int range_interp;
    int interp;

    bool noshift;
    bool nointerp;
    bool norange;
};

void apply_args(const struct st_xcorr_args *args, struct st_xcorr *xc);
void parse_opts(struct st_xcorr_args *xa, int argc, char **argv);

#endif
