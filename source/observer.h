#ifndef CONCAT_OBSERVER_H
#define CONCAT_OBSERVER_H

/* -----------------------------------------------------------------------
 * observer.h  -  Observation probability computation via KL-divergence NMF
 *
 * Mirrors observer.py::Observer.observe_np().
 *
 * For each particle, fit NMF activations H such that:
 *     W[:,particle_idxs] * H ≈ V  (KL divergence)
 * The observation log-likelihood is then:
 *     log p(V | particle) = -temperature * KL(V || W*H) / sum(V)
 * where KL(V||WH) = sum( V*log(V/WH) - V + WH )
 * ----------------------------------------------------------------------- */

#include "corpus.h"

typedef struct _observer {
    float temperature;  /* sharpness of observation model (default 50.0) */
    long  nmf_iters;    /* NMF iteration count (default 30) */
    long  sparsity;     /* activations per particle */
} t_observer;

/* Allocate. Returns NULL on failure. */
t_observer* observer_create(float temperature, long nmf_iters, long sparsity);

/* Free. */
void observer_free(t_observer *obs);

/* Compute log-observation weight for one particle.
 *   corpus      : loaded corpus (features, alpha)
 *   particle_idxs : sparsity indices into corpus frames
 *   V           : n_bins target spectrum (linear magnitudes)
 *   n_bins      : number of frequency bins
 * Returns the unnormalised log-weight. */
float observer_log_weight(t_observer *obs,
                          const t_corpus *corpus,
                          const int *particle_idxs,
                          const float *V,
                          long n_bins);

/* Compute log-weights for all N particles.
 *   all_idxs : N * sparsity row-major integer array
 *   log_weights : N output (caller allocates)
 *   V        : n_bins target spectrum
 *   N        : number of particles
 */
void observer_log_weights_all(t_observer *obs,
                               const t_corpus *corpus,
                               const int *all_idxs,
                               float *log_weights,
                               const float *V,
                               long N, long n_bins);

#endif /* CONCAT_OBSERVER_H */
