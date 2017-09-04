# Parallel xcorr program for GMTSAR

## Dependency

Multi-thread version (xcorr2) depends on following libraries:

- FFTW 3 (3.3.6 tested)
- Glib 2 (2.52.3 tested)

ArrayFire (GPGPU-accelerated) version (xcorr2) depends on following libraries:

- ArrayFire (3.5.0 tested)
- GPU drivers and GPGPU libraries (CUDA or OpenCL toolkits)

## Usage

xcorr2 and xcorr2_cl is designed to be seamlessly integrated into GMTSAR package.

To accelerate image registration in GMTSAR processing chain, you may:

- Simply replace original xcorr program with xcorr2 or xcorr2_cl
- Or, edit related CSH scripts
