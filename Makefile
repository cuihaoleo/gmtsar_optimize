CC = gcc
CXX = g++
INCLUDE = -I. `pkg-config --cflags glib-2.0 fftw3`
override CFLAGS := $(CFLAGS) $(INCLUDE) -std=gnu99
override CXXFLAGS := $(CXXFLAGS) $(INCLUDE) -std=c++11

all: xcorr2 xcorr2_cl
	rm -f *.o

array_helper.o: array_helper.c
	$(CC) -c -o $@ $^ $(CFLAGS)

fft_helper.o: fft_helper.c
	$(CC) -c -o $@ $^ $(CFLAGS)

prm_helper.o: prm_helper.c
	$(CC) -c -o $@ $^ $(CFLAGS)

xcorr2_args.o: xcorr2_args.c
	$(CC) -c -o $@ $^ $(CFLAGS)

xcorr2: xcorr2.c array_helper.o fft_helper.o prm_helper.o xcorr2_args.o
	$(CC) -o $@ $^ $(CFLAGS) -lm `pkg-config --libs glib-2.0 fftw3`

xcorr2_sg: xcorr2.c array_helper.o fft_helper.o prm_helper.o xcorr2_args.o
	$(CC) -o $@ $^ $(CFLAGS) -lm `pkg-config --libs glib-2.0 fftw3` -DNO_PTHREAD

xcorr2_cl: xcorr2_cl.cpp xcorr2_args.o prm_helper.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -laf `pkg-config --libs glib-2.0`

.PHONY: clean

clean:
	rm -f *.o
