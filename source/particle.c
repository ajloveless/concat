#include "particle.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* -----------------------------------------------------------------------
 * particle.c
 * ----------------------------------------------------------------------- */

/* -----------------------------------------------------------------------
 * Lifecycle
 * ----------------------------------------------------------------------- */

t_particle_filter* particle_filter_create(long N, long sparsity, long n_corpus,
                                           float pd, float temperature, long nmf_iters)
{
    t_particle_filter *pf = (t_particle_filter*)calloc(1, sizeof(t_particle_filter));
    if (!pf) return NULL;

    pf->N         = N;
    pf->sparsity  = sparsity;
    pf->n_corpus  = n_corpus;

    pf->all_idxs     = (int*)  malloc(N * sparsity * sizeof(int));
    pf->weights      = (float*)malloc(N * sizeof(float));
    pf->resample_buf = (int*)  malloc(N * sizeof(int));
    pf->chosen_idxs  = (int*)  malloc(sparsity * sizeof(int));

    pf->observer   = observer_create(temperature, nmf_iters, sparsity);
    pf->propagator = propagator_create(pd, n_corpus, sparsity);

    if (!pf->all_idxs || !pf->weights || !pf->resample_buf ||
        !pf->chosen_idxs || !pf->observer || !pf->propagator) {
        particle_filter_free(pf);
        return NULL;
    }

    particle_filter_reset(pf);
    return pf;
}

void particle_filter_free(t_particle_filter *pf)
{
    if (!pf) return;
    free(pf->all_idxs);
    free(pf->weights);
    free(pf->resample_buf);
    free(pf->chosen_idxs);
    observer_free(pf->observer);
    propagator_free(pf->propagator);
    free(pf);
}

void particle_filter_reset(t_particle_filter *pf)
{
    if (!pf) return;
    long N        = pf->N;
    long sparsity = pf->sparsity;
    long n_corpus = pf->n_corpus;

    /* Random uniform initialisation */
    for (long i = 0; i < N; i++) {
        for (long k = 0; k < sparsity; k++) {
            pf->all_idxs[i * sparsity + k] = (n_corpus > 0) ?
                (int)(rand() % n_corpus) : 0;
        }
        pf->weights[i] = 1.0f / N;
    }

    if (sparsity > 0 && n_corpus > 0) {
        for (long k = 0; k < sparsity; k++) {
            pf->chosen_idxs[k] = (int)(rand() % n_corpus);
        }
    }
}

void particle_filter_set_corpus_size(t_particle_filter *pf, long n_corpus)
{
    if (!pf) return;
    pf->n_corpus               = n_corpus;
    pf->propagator->n_corpus   = n_corpus;
    particle_filter_reset(pf);
}

/* -----------------------------------------------------------------------
 * Aggregate top particles by voting (mirrors aggregate_top_activations)
 *
 * Strategy:
 *  - Find the top-N/2 weighted particles
 *  - Collect all their activation indices (top-N/2 * sparsity values)
 *  - Vote by counting occurrences
 *  - Return the `sparsity` most-voted indices
 * ----------------------------------------------------------------------- */
void particle_aggregate_top(t_particle_filter *pf, int *out_idxs)
{
    long  N        = pf->N;
    long  sparsity = pf->sparsity;
    long  top_k    = N / 2;
    if (top_k < 1) top_k = 1;

    /* Find indices of top_k particles by weight */
    int *top_order = (int*)malloc(N * sizeof(int));
    if (!top_order) {
        /* Fallback: copy last chosen_idxs */
        memcpy(out_idxs, pf->chosen_idxs, sparsity * sizeof(int));
        return;
    }
    for (long i = 0; i < N; i++) top_order[i] = (int)i;

    /* Partial selection sort of top_k elements */
    for (long i = 0; i < top_k; i++) {
        long best = i;
        for (long j = i + 1; j < N; j++) {
            if (pf->weights[top_order[j]] > pf->weights[top_order[best]]) best = j;
        }
        int tmp = top_order[i]; top_order[i] = top_order[best]; top_order[best] = tmp;
    }

    /* Vote: count corpus index occurrences across top particles */
    long n_corpus = pf->n_corpus;
    int *votes = (int*)calloc(n_corpus, sizeof(int));
    if (!votes) {
        free(top_order);
        memcpy(out_idxs, pf->chosen_idxs, sparsity * sizeof(int));
        return;
    }

    for (long i = 0; i < top_k; i++) {
        long pidx = top_order[i];
        for (long k = 0; k < sparsity; k++) {
            int cidx = pf->all_idxs[pidx * sparsity + k];
            if (cidx >= 0 && cidx < (int)n_corpus) votes[cidx]++;
        }
    }

    /* Return `sparsity` highest-voted indices */
    for (long k = 0; k < sparsity; k++) {
        int best_idx = 0;
        int best_cnt = -1;
        for (long c = 0; c < n_corpus; c++) {
            if (votes[c] > best_cnt) { best_cnt = votes[c]; best_idx = (int)c; }
        }
        out_idxs[k] = best_idx;
        votes[best_idx] = -1; /* don't select again */
    }

    free(votes);
    free(top_order);
}

/* -----------------------------------------------------------------------
 * Main particle filter step  (intended to run in a worker thread)
 * ----------------------------------------------------------------------- */
int particle_filter_step(t_particle_filter *pf,
                          const t_corpus *corpus,
                          const float *V,
                          long n_bins)
{
    if (!pf || !corpus || !V) return -1;
    if (corpus->n_frames == 0) return -1;

    long N        = pf->N;
    long sparsity = pf->sparsity;

    /* 1. Propagate */
    propagator_step_all(pf->propagator, pf->all_idxs, N);

    /* 2. Compute log-observation weights */
    float *log_weights = (float*)malloc(N * sizeof(float));
    if (!log_weights) return -1;

    observer_log_weights_all(pf->observer,
                              corpus,
                              pf->all_idxs,
                              log_weights,
                              V, N, n_bins);

    /* 3. Combine with prior weights (log domain) */
    for (long i = 0; i < N; i++) {
        pf->weights[i] = logf((pf->weights[i] > 0.0f) ? pf->weights[i] : 1e-30f)
                         + log_weights[i];
    }
    free(log_weights);

    /* 4. Softmax-normalise */
    utils_normalise_weights(pf->weights, N);

    /* 5. Aggregate top activations → chosen_idxs */
    particle_aggregate_top(pf, pf->chosen_idxs);

    /* 6. Resample if ESS too low */
    float ess = utils_stochastic_resample(pf->weights, pf->resample_buf, N);
    if (ess < PARTICLE_ESS_THRESHOLD * N) {
        /* Rebuild all_idxs from resampled indices */
        int *new_idxs = (int*)malloc(N * sparsity * sizeof(int));
        if (new_idxs) {
            for (long i = 0; i < N; i++) {
                int src = pf->resample_buf[i];
                memcpy(new_idxs + i * sparsity,
                       pf->all_idxs + src * sparsity,
                       sparsity * sizeof(int));
            }
            memcpy(pf->all_idxs, new_idxs, N * sparsity * sizeof(int));
            free(new_idxs);
            /* Weights already normalised by stochastic_resample */
        }
    }

    return 0;
}
