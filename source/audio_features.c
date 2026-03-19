#include "audio_features.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * features.c
 * ----------------------------------------------------------------------- */

/* -----------------------------------------------------------------------
 * Cooley-Tukey iterative FFT (DIT, radix-2).
 * buf : 2*n floats (interleaved re/im).
 * n   : must be power of 2.
 * ----------------------------------------------------------------------- */
static void fft_complex(float *buf, long n)
{
    /* Bit-reversal permutation */
    for (long i = 1, j = 0; i < n; i++) {
        long bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = buf[2*i]; buf[2*i] = buf[2*j]; buf[2*j] = tr;
            float ti = buf[2*i+1]; buf[2*i+1] = buf[2*j+1]; buf[2*j+1] = ti;
        }
    }

    /* Butterfly stages */
    for (long len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * 3.14159265358979323846f / (float)len;
        float wr = cosf(ang), wi = sinf(ang);
        for (long i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (long j = 0; j < len / 2; j++) {
                long u = 2*(i + j);
                long v = 2*(i + j + len/2);
                float ur = buf[u], ui = buf[u+1];
                float vr = buf[v]*cr - buf[v+1]*ci;
                float vi = buf[v]*ci + buf[v+1]*cr;
                buf[u]   = ur + vr;
                buf[u+1] = ui + vi;
                buf[v]   = ur - vr;
                buf[v+1] = ui - vi;
                float ncr = cr*wr - ci*wi;
                ci = cr*wi + ci*wr;
                cr = ncr;
            }
        }
    }
}

/* Public real FFT: pack real samples into complex array, run FFT */
void fft_real(float *buf, long n)
{
    /* buf already has imaginary parts zeroed */
    fft_complex(buf, n);
    /* Only bins 0..n/2 are meaningful (Hermitian symmetry) */
}

/* -----------------------------------------------------------------------
 * Feature computer
 * ----------------------------------------------------------------------- */

t_feature_computer* features_create(long win)
{
    t_feature_computer *fc = (t_feature_computer*)calloc(1, sizeof(t_feature_computer));
    if (!fc) return NULL;

    fc->win    = win;
    fc->hop    = win / 2;
    fc->n_bins = win / 2 + 1;

    fc->window  = (float*)malloc(win * sizeof(float));
    fc->fft_buf = (float*)malloc(2 * win * sizeof(float));

    if (!fc->window || !fc->fft_buf) {
        features_free(fc);
        return NULL;
    }

    /* Pre-compute Hann window */
    for (long i = 0; i < win; i++) {
        fc->window[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979323846f * i / (win - 1)));
    }

    return fc;
}

void features_free(t_feature_computer *fc)
{
    if (!fc) return;
    free(fc->window);
    free(fc->fft_buf);
    free(fc);
}

void features_compute(t_feature_computer *fc, const float *in, float *out)
{
    long win = fc->win;

    /* Zero imaginary half */
    memset(fc->fft_buf, 0, 2 * win * sizeof(float));

    /* Copy windowed samples into real part */
    for (long i = 0; i < win; i++) {
        fc->fft_buf[2 * i]     = in[i] * fc->window[i];
        fc->fft_buf[2 * i + 1] = 0.0f;
    }

    fft_real(fc->fft_buf, win);

    /* Extract magnitudes for bins 0..n_bins-1 */
    for (long k = 0; k < fc->n_bins; k++) {
        float re = fc->fft_buf[2 * k];
        float im = fc->fft_buf[2 * k + 1];
        out[k] = sqrtf(re * re + im * im);
    }
}

void features_compute_db(t_feature_computer *fc, const float *in, float *out)
{
    features_compute(fc, in, out);
    utils_to_db(out, fc->n_bins);
}
