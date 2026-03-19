#include "observer.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* -----------------------------------------------------------------------
 * observer.c
 * ----------------------------------------------------------------------- */

t_observer* observer_create(float temperature, long nmf_iters, long sparsity)
{
    t_observer *o = (t_observer*)calloc(1, sizeof(t_observer));
    if (!o) return NULL;
    o->temperature = temperature;
    o->nmf_iters   = nmf_iters;
    o->sparsity    = sparsity;
    return o;
}

void observer_free(t_observer *o)
{
    free(o);
}

/* -----------------------------------------------------------------------
 * KL divergence: D_KL(V || WH) = sum( V*log(V/WH) - V + WH )
 *
 * Note: we use linear-domain magnitudes as in the Python reference.
 * ----------------------------------------------------------------------- */
static float kl_divergence(const float *V, const float *WH, long M)
{
    float kl = 0.0f;
    for (long m = 0; m < M; m++) {
        float v  = V[m];
        float wh = (WH[m] > 1e-10f) ? WH[m] : 1e-10f;
        if (v > 1e-10f) kl += v * logf(v / wh) - v + wh;
        else            kl += wh;
    }
    return kl;
}

/* -----------------------------------------------------------------------
 * Compute log-observation weight for one particle.
 * ----------------------------------------------------------------------- */
float observer_log_weight(t_observer *obs,
                          const t_corpus *corpus,
                          const int *particle_idxs,
                          const float *V,
                          long n_bins)
{
    long   M        = n_bins;
    long   sparsity = obs->sparsity;
    float *H        = utils_nmf_fit(corpus->features,
                                    corpus->alpha,
                                    particle_idxs,
                                    sparsity,
                                    V, M,
                                    obs->nmf_iters);
    if (!H) return -1e30f;

    /* Reconstruct WH = W[:,idxs] * H */
    float *WH = (float*)calloc(M, sizeof(float));
    if (!WH) { free(H); return -1e30f; }

    for (long k = 0; k < sparsity; k++) {
        const float *Wk = corpus->features + (long)particle_idxs[k] * M;
        float hk = H[k];
        for (long m = 0; m < M; m++) WH[m] += Wk[m] * hk;
    }

    /* Normalise V for KL (so result is independent of overall level) */
    float V_sum = 0.0f;
    for (long m = 0; m < M; m++) V_sum += V[m];
    if (V_sum < 1e-10f) V_sum = 1.0f;

    float kl   = kl_divergence(V, WH, M);
    float logw = -obs->temperature * kl / V_sum;

    free(H);
    free(WH);
    return logw;
}

/* -----------------------------------------------------------------------
 * Batch computation for all N particles.
 * ----------------------------------------------------------------------- */
void observer_log_weights_all(t_observer *obs,
                               const t_corpus *corpus,
                               const int *all_idxs,
                               float *log_weights,
                               const float *V,
                               long N, long n_bins)
{
    for (long i = 0; i < N; i++) {
        log_weights[i] = observer_log_weight(obs,
                                              corpus,
                                              all_idxs + i * obs->sparsity,
                                              V, n_bins);
    }
}
