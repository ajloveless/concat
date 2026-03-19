#ifndef CONCAT_PARTICLE_H
#define CONCAT_PARTICLE_H

/* -----------------------------------------------------------------------
 * particle.h  -  Particle filter for concatenative synthesis
 *
 * Mirrors particle.py::ParticleFilterChannel
 *
 * Owns:
 *   - N particles, each holding `sparsity` corpus-frame indices
 *   - Particle weights
 *   - Observer and Propagator
 *
 * Exposes two entry points:
 *   particle_filter_step()  - the heavy computation (runs in worker thread)
 *   particle_get_chosen()   - called after step to retrieve winning indices
 * ----------------------------------------------------------------------- */

#include "corpus.h"
#include "observer.h"
#include "propagator.h"

/* Effective sample size threshold below which we resample */
#define PARTICLE_ESS_THRESHOLD 0.5f

typedef struct _particle_filter {
    long  N;                /* number of particles */
    long  sparsity;         /* activations per particle */
    long  n_corpus;         /* cache of corpus->n_frames */

    /* all_idxs[i * sparsity + k] = k-th index for particle i */
    int  *all_idxs;         /* N * sparsity */
    float *weights;          /* N unnormalised weights */
    int  *resample_buf;     /* N scratch buffer for resampling */

    t_observer   *observer;
    t_propagator *propagator;

    /* Last chosen indices (output of particle_filter_step) */
    int  *chosen_idxs;      /* sparsity */
} t_particle_filter;

/* -----------------------------------------------------------------------
 * Lifecycle
 * ----------------------------------------------------------------------- */

/* Create particle filter.
 *   N         : number of particles
 *   sparsity  : activations per particle
 *   n_corpus  : number of corpus frames (for random initialisation)
 *   pd        : temporal continuity probability
 *   temperature: observation sharpness
 *   nmf_iters : NMF iterations per particle
 * Returns NULL on failure. */
t_particle_filter* particle_filter_create(long N, long sparsity, long n_corpus,
                                           float pd, float temperature, long nmf_iters);

/* Free all resources. */
void particle_filter_free(t_particle_filter *pf);

/* Reset to uniform random state. */
void particle_filter_reset(t_particle_filter *pf);

/* Update corpus size after a new corpus is loaded. */
void particle_filter_set_corpus_size(t_particle_filter *pf, long n_corpus);

/* -----------------------------------------------------------------------
 * Processing  (designed to run in a worker thread)
 * ----------------------------------------------------------------------- */

/* Run one full particle filter step:
 *   1. Propagate all particles
 *   2. Compute observation weights
 *   3. Aggregate top activations → chosen_idxs
 *   4. Resample if needed
 *
 *   corpus  : loaded corpus (features & audio)
 *   V       : n_bins target spectrum (linear magnitudes)
 *   n_bins  : number of frequency bins
 *
 * Results are stored in pf->chosen_idxs.
 * Returns 0 on success, -1 on error.
 */
int particle_filter_step(t_particle_filter *pf,
                          const t_corpus *corpus,
                          const float *V,
                          long n_bins);

/* -----------------------------------------------------------------------
 * Helpers
 * ----------------------------------------------------------------------- */

/* Aggregate the top-weighted particles and return a consensus set of
 * `sparsity` corpus-frame indices using voting (mirrors
 * particle.py::aggregate_top_activations).
 *
 * Results written into out_idxs[0..sparsity-1]. */
void particle_aggregate_top(t_particle_filter *pf, int *out_idxs);

#endif /* CONCAT_PARTICLE_H */
