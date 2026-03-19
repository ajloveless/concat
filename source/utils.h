#ifndef CONCAT_UTILS_H
#define CONCAT_UTILS_H

#include <stddef.h>

/* -----------------------------------------------------------------------
 * utils.h - NMF multiplicative updates, stochastic resampling, math helpers
 * ----------------------------------------------------------------------- */

/* Hann window applied in-place to buf[0..n-1] */
void utils_hann_window(float *buf, long n);

/* Compute power spectrum magnitude from complex interleaved FFT output.
 * fft_out has (n/2+1)*2 floats (real, imag pairs).
 * out receives (n/2+1) magnitude values. */
void utils_magnitude_spectrum(const float *fft_out, long n, float *out);

/* Convert linear magnitudes to dB power (20*log10, floor at -120 dB) */
void utils_to_db(float *buf, long n);

/* KL-divergence NMF multiplicative H update.
 *   W   : M x K  (corpus features, column-major: W[k*M + m])
 *   V   : M      (target spectrum)
 *   H   : K      (activations, updated in-place)
 *   alpha : regularisation weight
 *   M, K, iters
 */
void utils_nmf_update(const float *W, const float *V,
                      float *H, float alpha,
                      long M, long K, long iters);

/* Fit NMF activations for a sparse subset of corpus frames.
 *   W_full  : M x N_corpus (column-major)
 *   alpha   : regularisation
 *   idxs    : sparsity indices into corpus columns
 *   sparsity: length of idxs
 *   V       : M target spectrum
 *   iters   : NMF iterations
 *   Returns newly allocated float[sparsity] activations (caller frees). */
float* utils_nmf_fit(const float *W_full, float alpha,
                     const int *idxs, long sparsity,
                     const float *V, long M, long iters);

/* Synthesize one output window via weighted overlap-add.
 *   corpus_audio : win x N_corpus (column-major)
 *   idxs         : sparsity indices
 *   activations  : sparsity weights
 *   sparsity     : number of active components
 *   out          : win output samples (zeroed before call)
 *   win          : window size
 */
void utils_synthesize_audio(const float *corpus_audio,
                             const int *idxs,
                             const float *activations,
                             long sparsity,
                             float *out, long win);

/* Stochastic universal resampling.
 *   weights : N (unnormalised, modified in place → normalised)
 *   indices : N output integer array (resampled indices)
 *   N       : number of particles
 * Returns effective sample size (1/sum(w^2)). */
float utils_stochastic_resample(float *weights, int *indices, long N);

/* Softmax-normalise weights in place, returns max weight. */
float utils_normalise_weights(float *weights, long N);

/* Safe log (returns -1e30 for non-positive inputs) */
float utils_safe_log(float x);

/* Simple dot product */
float utils_dot(const float *a, const float *b, long n);

#endif /* CONCAT_UTILS_H */
