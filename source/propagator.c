#include "propagator.h"
#include <stdlib.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * propagator.c
 * ----------------------------------------------------------------------- */

t_propagator* propagator_create(float pd, long n_corpus, long sparsity)
{
    t_propagator *p = (t_propagator*)calloc(1, sizeof(t_propagator));
    if (!p) return NULL;
    p->pd       = pd;
    p->n_corpus = n_corpus;
    p->sparsity = sparsity;
    return p;
}

void propagator_free(t_propagator *p)
{
    free(p);
}

/* -----------------------------------------------------------------------
 * Propagate a single particle's activation indices in place.
 *
 * Mirrors propagator.py::Propagator.propagate():
 *   for each activation index:
 *     u = rand()
 *     if u < pd  →  advance by 1 (wrap at n_corpus boundary)
 *     else       →  jump to random position in corpus
 * ----------------------------------------------------------------------- */
void propagator_step(t_propagator *p, int *particle_idxs)
{
    for (long k = 0; k < p->sparsity; k++) {
        float u = (float)rand() / ((float)RAND_MAX + 1.0f);
        if (u < p->pd) {
            /* Temporal continuity: advance one frame */
            int next = particle_idxs[k] + 1;
            if (next >= (int)p->n_corpus) next = 0;
            particle_idxs[k] = next;
        } else {
            /* Random jump: draw a fresh uniform sample mapped to [0, n_corpus) */
            float r = (float)rand() / ((float)RAND_MAX + 1.0f);
            int idx = (int)(r * p->n_corpus);
            if (idx >= (int)p->n_corpus) idx = (int)p->n_corpus - 1;
            if (idx < 0)                 idx = 0;
            particle_idxs[k] = idx;
        }
    }
}

void propagator_step_all(t_propagator *p, int *all_idxs, long N)
{
    for (long i = 0; i < N; i++) {
        propagator_step(p, all_idxs + i * p->sparsity);
    }
}
