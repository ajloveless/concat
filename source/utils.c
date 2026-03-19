#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* -----------------------------------------------------------------------
 * utils.c
 * ----------------------------------------------------------------------- */

/* Hann window */
void utils_hann_window(float *buf, long n)
{
    for (long i = 0; i < n; i++) {
        float w = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979323846f * i / (n - 1)));
        buf[i] *= w;
    }
}

/* Magnitude spectrum from interleaved complex FFT */
void utils_magnitude_spectrum(const float *fft_out, long n, float *out)
{
    long bins = n / 2 + 1;
    for (long k = 0; k < bins; k++) {
        float re = fft_out[2 * k];
        float im = fft_out[2 * k + 1];
        out[k] = sqrtf(re * re + im * im);
    }
}

/* Linear → dB power */
void utils_to_db(float *buf, long n)
{
    for (long i = 0; i < n; i++) {
        float v = buf[i];
        buf[i] = (v > 1e-6f) ? 20.0f * log10f(v) : -120.0f;
    }
}

/* Safe log */
float utils_safe_log(float x)
{
    return (x > 0.0f) ? logf(x) : -1e30f;
}

/* Dot product */
float utils_dot(const float *a, const float *b, long n)
{
    float s = 0.0f;
    for (long i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* -----------------------------------------------------------------------
 * KL-divergence NMF: multiplicative H update.
 *
 * Minimises D_KL(V || W*H) w.r.t. H with L1 regularisation.
 * Update rule (from probutils.py::do_KL):
 *   H *= (W^T * (V / (W*H))) / (col_sum(W) + alpha)
 *
 * W is M x K, column-major: element (m, k) = W[k*M + m]
 * ----------------------------------------------------------------------- */
void utils_nmf_update(const float *W, const float *V,
                      float *H, float alpha,
                      long M, long K, long iters)
{
    float *WH    = (float*)malloc(M * sizeof(float));
    float *ratio = (float*)malloc(M * sizeof(float));
    if (!WH || !ratio) { free(WH); free(ratio); return; }

    /* Pre-compute column sums of W */
    float *Wsum = (float*)malloc(K * sizeof(float));
    if (!Wsum) { free(WH); free(ratio); return; }
    for (long k = 0; k < K; k++) {
        float s = 0.0f;
        const float *Wk = W + k * M;
        for (long m = 0; m < M; m++) s += Wk[m];
        Wsum[k] = s;
    }

    for (long iter = 0; iter < iters; iter++) {
        /* WH = W * H  (M-vector) */
        for (long m = 0; m < M; m++) {
            float s = 0.0f;
            for (long k = 0; k < K; k++) s += W[k * M + m] * H[k];
            WH[m] = (s > 1e-10f) ? s : 1e-10f;
        }

        /* ratio = V / WH */
        for (long m = 0; m < M; m++) ratio[m] = V[m] / WH[m];

        /* H[k] *= (W[:,k]^T * ratio) / (Wsum[k] + alpha * H[k]) */
        for (long k = 0; k < K; k++) {
            const float *Wk = W + k * M;
            float num = 0.0f;
            for (long m = 0; m < M; m++) num += Wk[m] * ratio[m];
            float denom = Wsum[k] + alpha * H[k];
            if (denom < 1e-10f) denom = 1e-10f;
            H[k] = H[k] * num / denom;
            if (H[k] < 0.0f) H[k] = 0.0f;
        }
    }

    free(WH); free(ratio); free(Wsum);
}

/* -----------------------------------------------------------------------
 * Fit NMF activations for a sparse subset of corpus frames.
 * ----------------------------------------------------------------------- */
float* utils_nmf_fit(const float *W_full, float alpha,
                     const int *idxs, long sparsity,
                     const float *V, long M, long iters)
{
    /* Build sub-matrix W_sub : M x sparsity */
    float *W_sub = (float*)malloc(M * sparsity * sizeof(float));
    float *H     = (float*)malloc(sparsity * sizeof(float));
    if (!W_sub || !H) { free(W_sub); free(H); return NULL; }

    for (long k = 0; k < sparsity; k++) {
        memcpy(W_sub + k * M, W_full + idxs[k] * M, M * sizeof(float));
        H[k] = 1.0f;   /* uniform initialisation */
    }

    utils_nmf_update(W_sub, V, H, alpha, M, sparsity, iters);

    free(W_sub);
    return H;
}

/* -----------------------------------------------------------------------
 * Synthesize one output window.
 * ----------------------------------------------------------------------- */
void utils_synthesize_audio(const float *corpus_audio,
                             const int *idxs,
                             const float *activations,
                             long sparsity,
                             float *out, long win)
{
    memset(out, 0, win * sizeof(float));
    for (long k = 0; k < sparsity; k++) {
        const float *src = corpus_audio + (long)idxs[k] * win;
        float a = activations[k];
        for (long i = 0; i < win; i++) out[i] += a * src[i];
    }
}

/* -----------------------------------------------------------------------
 * Stochastic universal resampling (see probutils.py::stochastic_universal_sample)
 * ----------------------------------------------------------------------- */
float utils_stochastic_resample(float *weights, int *indices, long N)
{
    /* Normalise */
    float sum = 0.0f;
    for (long i = 0; i < N; i++) sum += weights[i];
    if (sum < 1e-30f) {
        /* Degenerate: uniform reset */
        for (long i = 0; i < N; i++) { weights[i] = 1.0f / N; indices[i] = (int)i; }
        return (float)N;
    }
    float inv_sum = 1.0f / sum;
    for (long i = 0; i < N; i++) weights[i] *= inv_sum;

    /* Compute ESS = 1 / sum(w^2) */
    float ess_denom = 0.0f;
    for (long i = 0; i < N; i++) ess_denom += weights[i] * weights[i];
    float ess = (ess_denom > 0.0f) ? 1.0f / ess_denom : (float)N;

    /* Build CDF and sample */
    float *cdf = (float*)malloc((N + 1) * sizeof(float));
    if (!cdf) return ess;
    cdf[0] = 0.0f;
    for (long i = 0; i < N; i++) cdf[i + 1] = cdf[i] + weights[i];

    float step  = 1.0f / N;
    float start = step * ((float)rand() / ((float)RAND_MAX + 1.0f));

    long j = 0;
    for (long i = 0; i < N; i++) {
        float u = start + i * step;
        while (j < N - 1 && cdf[j + 1] < u) j++;
        indices[i] = (int)j;
    }

    free(cdf);
    return ess;
}

/* -----------------------------------------------------------------------
 * Softmax-normalise weights in place.
 * ----------------------------------------------------------------------- */
float utils_normalise_weights(float *weights, long N)
{
    float mx = -FLT_MAX;
    for (long i = 0; i < N; i++) if (weights[i] > mx) mx = weights[i];

    float sum = 0.0f;
    for (long i = 0; i < N; i++) {
        weights[i] = expf(weights[i] - mx);
        sum += weights[i];
    }
    if (sum < 1e-30f) sum = 1e-30f;
    float inv = 1.0f / sum;
    for (long i = 0; i < N; i++) weights[i] *= inv;

    return mx;
}
