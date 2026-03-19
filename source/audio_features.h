#ifndef CONCAT_FEATURES_H
#define CONCAT_FEATURES_H

/* -----------------------------------------------------------------------
 * features.h  -  FFT and spectral feature extraction
 *
 * Implements a real-valued FFT using the split-radix Cooley-Tukey algorithm
 * and provides the AudioFeatureComputer abstraction that mirrors
 * audiofeatures.py::AudioFeatureComputer.
 * ----------------------------------------------------------------------- */

/* -----------------------------------------------------------------------
 * Low-level real FFT
 * ----------------------------------------------------------------------- */

/* In-place real FFT.  buf must have 2*n floats (interleaved re/im).
 * On entry:  buf[0..n-1] = real samples, buf[n..2n-1] = 0.
 * On exit:   buf[2k], buf[2k+1] = Re, Im of bin k, for k = 0..n/2.
 * n must be a power of 2. */
void fft_real(float *buf, long n);

/* -----------------------------------------------------------------------
 * Feature computer
 * ----------------------------------------------------------------------- */

typedef struct _feature_computer {
    long   win;        /* window size (power of 2) */
    long   hop;        /* hop size (win/2) */
    long   n_bins;     /* number of frequency bins = win/2+1 */
    float *window;     /* pre-computed Hann window coefficients */
    float *fft_buf;    /* scratch: 2*win floats */
} t_feature_computer;

/* Create a feature computer for the given window size.
 * Returns NULL on allocation failure. */
t_feature_computer* features_create(long win);

/* Free all resources. */
void features_free(t_feature_computer *fc);

/* Compute the magnitude spectrum of one audio frame.
 *   in   : win input samples
 *   out  : n_bins magnitude values (caller allocates)
 *
 * Applies Hann window, computes RFFT, returns magnitudes. */
void features_compute(t_feature_computer *fc, const float *in, float *out);

/* Same as features_compute but converts magnitudes to dB power. */
void features_compute_db(t_feature_computer *fc, const float *in, float *out);

#endif /* CONCAT_FEATURES_H */
