#include "xcorr2.h"
#include "xcorr2_args.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <ctype.h>

void strtrim(char *s) {
    char *p = s;

    while (*p) {
        if (isspace(*p)) {
            p++;
            continue;
        }

        *(s++) = *(p++);
    }

    *s = *p;
}

void apply_args(const struct st_xcorr_args *args, struct st_xcorr *xc) {
    int num_patches, num_valid_az;
    double prf[2];
    struct prm_handler m_prm = prm_open(args->m_prm);
    struct prm_handler s_prm = prm_open(args->s_prm);

    xc->m_path = strdup(prm_get_str(m_prm, "SLC_file"));
    strtrim(xc->m_path);
    xc->s_path = strdup(prm_get_str(s_prm, "SLC_file"));
    strtrim(xc->s_path);

    xc->m_nx = prm_get_int(m_prm, "num_rng_bins");
    num_patches = prm_get_int(m_prm, "num_patches");
    num_valid_az = prm_get_int(m_prm, "num_valid_az");
    xc->m_ny = num_patches * num_valid_az;

    xc->s_nx = prm_get_int(s_prm, "num_rng_bins");
    num_patches = prm_get_int(s_prm, "num_patches");
    num_valid_az = prm_get_int(s_prm, "num_valid_az");
    xc->s_ny = num_patches * num_valid_az;

    prf[0] = prm_get_f64(m_prm, "PRF");
    prf[1] = prm_get_f64(s_prm, "PRF");
    xc->astretcha = prf[0] > 0 ? (prf[1] - prf[0]) / prf[0] : 0.0;

    if (!args->noshift) {
        xc->x_offset = prm_get_int(s_prm, "rshift");
        xc->y_offset = prm_get_int(s_prm, "ashift");
    } else
        xc->x_offset = xc->y_offset = 0;

    xc->xsearch = args->xsearch ? args->xsearch : 64;
    xc->ysearch = args->ysearch ? args->ysearch : 64;

    xc->nxl = args->nx ? args->nx : 16;
    xc->nyl = args->ny ? args->ny : 32;

    if (args->norange)
        xc->ri = 1;
    else
        xc->ri = args->range_interp ? args->range_interp : 2;

    if (args->nointerp)
        xc->interp_factor = 1;
    else
        xc->interp_factor = args->interp ? args->interp : 16;

    xc->n2x = xc->n2y = 8;

    prm_close(&m_prm);
    prm_close(&s_prm);
}

void parse_opts(struct st_xcorr_args *xa, int argc, char **argv) {
    enum {
        OPT_NX = 10,
        OPT_NY = 20,
        OPT_RANGE_INTERP = 30,
        OPT_XSEARCH = 40,
        OPT_YSEARCH = 50,
        OPT_INTERP = 60,
        OPT_HELP = 100,
        OPT_NO_SHIFT = -10,
        OPT_NOINTERP = -20,
        OPT_NORANGE = -30,
        OPT_DEVICE = -40,
    };

    static const char *help = \
        "xcorr [GMT5SAR] - Compute 2-D cross-correlation of two images\n\n\n"
        "Usage: xcorr master.PRM slave.PRM [-nx n] [-ny n] [-xsearch xs] [-ysearch ys]\n"
        "master.PRM             PRM file for reference image\n"
        "slave.PRM              PRM file of secondary image\n"
        "-noshift               ignore ashift and rshift in prm file (set to 0)\n"
        "-nx  nx                number of locations in x (range) direction (int)\n"
        "-ny  ny                number of locations in y (azimuth) direction (int)\n"
        "-nointerp              do not interpolate correlation function\n"
        "-range_interp ri       interpolate range by ri (power of two) [default: 2]\n"
        "-norange               do not range interpolate \n"
        "-xsearch xs            search window size in x (range) direction (int power of 2 [32 64 128 256])\n"
        "-ysearch ys            search window size in y (azimuth) direction (int power of 2 [32 64 128 256])\n"
        "-interp  factor        interpolate correlation function by factor (int) [default, 16]\n"
        "-af [cuda|opencl|cpu]  ArrayFire accelerate backend \n"
        "output: \n freq_xcorr.dat (default) \n time_xcorr.dat (if -time option))\n"
        "\nuse fitoffset.csh to convert output to PRM format\n"
        "\nExample:\n"
        "xcorr IMG-HH-ALPSRP075880660-H1.0__A.PRM IMG-HH-ALPSRP129560660-H1.0__A.PRM -nx 20 -ny 50 \n";

    static struct option long_options[] = {
        { "noshift", no_argument, NULL, OPT_NO_SHIFT },
        { "nx", required_argument, NULL, OPT_NX },
        { "ny", required_argument, NULL, OPT_NY },
        { "nointerp", no_argument, NULL, OPT_NOINTERP },
        { "norange", no_argument, NULL, OPT_NORANGE },
        { "range_interp", required_argument, NULL, OPT_RANGE_INTERP },
        { "xsearch", required_argument, NULL, OPT_XSEARCH },
        { "ysearch", required_argument, NULL, OPT_YSEARCH },
        { "interp", required_argument, NULL, OPT_INTERP },
        { "af", required_argument, NULL, OPT_DEVICE },
        { "help", required_argument, NULL, OPT_HELP },
        { 0, 0, 0, 0 },
    };

    if (argc == 1) {
        fputs(help, stdout);
        exit(0);
    }

    memset(xa, 0, sizeof(struct st_xcorr_args));

    while (1) {
        int opt, long_index, int_arg;
        opt = getopt_long_only(argc, argv, "", long_options, &long_index);

        if (opt == -1) break;

        if (opt > 0) {
            char *endptr;
            int_arg = strtol(optarg, &endptr, 10);
            if (*endptr != '\0') {
                fprintf(stderr, "Invalid argument for -%s option", long_options[long_index].name);
                exit(-1);
            }
        }

        switch (opt) {
            case OPT_NX:
                xa->nx = int_arg;
                break;
            case OPT_NY:
                xa->ny = int_arg;
                break;
            case OPT_XSEARCH:
                xa->xsearch = int_arg;
                break;
            case OPT_YSEARCH:
                xa->ysearch = int_arg;
                break;
            case OPT_INTERP:
                xa->interp = int_arg;
                break;
            case OPT_DEVICE:
                if (!strcmp(optarg, "cuda"))
                    xa->device = XCORR2_DEVICE_CUDA;
                else if (!strcmp(optarg, "opencl"))
                    xa->device = XCORR2_DEVICE_OPENCL;
                else if (!strcmp(optarg, "cpu"))
                    xa->device = XCORR2_DEVICE_CPU;
                else {
                    fputs(help, stdout);
                    exit(-1);
                }
                break;
            case OPT_RANGE_INTERP:
                xa->range_interp = int_arg;
                break;
            case OPT_NO_SHIFT:
                xa->noshift = true;
                break;
            case OPT_NOINTERP:
                xa->nointerp = true;
                break;
            case OPT_NORANGE:
                xa->norange = true;
                break;
            case OPT_HELP:
                fputs(help, stdout);
                exit(0);
            default:
                abort();
        }
    }

    if (optind < argc)
        xa->m_prm = argv[optind++];
    else {
        fprintf(stderr, "PRM file of master not specified\n");
        exit(-1);
    }

    if (optind < argc)
        xa->s_prm = argv[optind++];
    else {
        fprintf(stderr, "PRM file of slave not specified\n");
        exit(-1);
    }
}
