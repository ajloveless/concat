#ifndef CONCAT_PROPAGATOR_H
#define CONCAT_PROPAGATOR_H

/* -----------------------------------------------------------------------
 * propagator.h  -  Particle state transition model
 *
 * Mirrors propagator.py::Propagator.propagate().
 *
 * Each particle holds `sparsity` integer corpus-frame indices.  At each
 * step, each index either:
 *   - advances by 1 (stays in temporal sequence)  with probability pd
 *   - jumps to a random corpus frame               with probability 1-pd
 * ----------------------------------------------------------------------- */

typedef struct _propagator {
    float pd;           /* temporal continuity probability (default 0.95) */
    long  n_corpus;     /* number of corpus frames */
    long  sparsity;     /* number of activations per particle */
} t_propagator;

/* Allocate and initialise a propagator.  Returns NULL on failure. */
t_propagator* propagator_create(float pd, long n_corpus, long sparsity);

/* Free. */
void propagator_free(t_propagator *p);

/* Propagate one particle in place.
 *   particle_idxs : sparsity integers, each in [0, n_corpus-1].
 * Modifies the array in place. */
void propagator_step(t_propagator *p, int *particle_idxs);

/* Propagate all N particles.
 *   all_idxs : N * sparsity contiguous integers (row-major per particle).
 *   N        : number of particles. */
void propagator_step_all(t_propagator *p, int *all_idxs, long N);

#endif /* CONCAT_PROPAGATOR_H */
