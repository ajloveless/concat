/* -----------------------------------------------------------------------
 * concat_task.c  -  Thread pool worker, completion callback, submission
 * ----------------------------------------------------------------------- */

#include "concat_task.h"
#include "concat_tilde.h"    /* t_concat definition (no circular dependency) */
#include "concat_buffer.h"
#include "particle.h"
#include "audio_features.h"
#include "utils.h"

#include "ext.h"
#include "ext_obex.h"
#include "ext_atomic.h"
#include "ext_systhread.h"
#include "ext_systime.h"

#include <string.h>

/* -----------------------------------------------------------------------
 * Worker: runs in a background thread from the Max thread pool.
 * ----------------------------------------------------------------------- */
void concat_task_worker(t_concat_task_args *args)
{
    t_concat *x = args->owner;

    systhread_mutex_lock(x->corpus_mutex);

    if (!x->corpus_loaded || !x->corpus || x->corpus->n_frames == 0) {
        systhread_mutex_unlock(x->corpus_mutex);
        return;
    }

    long write_idx = atom_getlong(&x->write_idx);
    t_concat_output_buffer *write_buf = x->output_bufs[write_idx];

    /* Mark write buffer as being written */
    atom_setlong(&write_buf->ready, 0);

    long n_bins   = x->n_bins;
    long win      = x->win;
    long sparsity = x->sparsity;

    /* 1. Run particle filter step (propagate + observe + resample) */
    int ret = particle_filter_step(x->pf, x->corpus,
                                    args->input_features, n_bins);

    if (ret == 0) {
        /* 2. Fit NMF activations for the chosen corpus indices */
        float *activations = utils_nmf_fit(x->corpus->features,
                                            x->corpus->alpha,
                                            x->pf->chosen_idxs,
                                            sparsity,
                                            args->input_features,
                                            n_bins,
                                            x->nmf_iters);
        if (activations) {
            /* 3. Synthesize audio into write buffer */
            utils_synthesize_audio(x->corpus->audio,
                                   x->pf->chosen_idxs,
                                   activations,
                                   sparsity,
                                   write_buf->audio,
                                   win);
            free(activations);   /* utils_nmf_fit uses malloc */

            write_buf->hop_offset = 0;
            atom_setlong(&write_buf->ready, 1);
        }
    }

    systhread_mutex_unlock(x->corpus_mutex);
}

/* -----------------------------------------------------------------------
 * Completion callback: background thread — cannot call outlets.
 * ----------------------------------------------------------------------- */
void concat_task_complete(t_concat_task_args *args)
{
    /* Defer buffer swap to main thread */
    defer_low(args->owner, (method)concat_task_finish, NULL, 0, NULL);

    /* Free per-task resources */
    sysmem_freeptr(args->input_features);
    sysmem_freeptr(args);
}

/* -----------------------------------------------------------------------
 * Main-thread callback: safe to modify shared state and send to outlets.
 * ----------------------------------------------------------------------- */
void concat_task_finish(t_concat *x, t_symbol *sym, long argc, void *argv)
{
    /* Atomic swap: read ↔ write buffer */
    long old_read  = atom_getlong(&x->read_idx);
    long old_write = atom_getlong(&x->write_idx);
    atom_setlong(&x->read_idx,  old_write);
    atom_setlong(&x->write_idx, old_read);

    /* Release processing flag */
    atom_setlong(&x->processing_flag, 0);

    x->frames_completed++;

    /* Report stats via info outlet */
    t_atom stats[2];
    atom_setlong(&stats[0], x->frames_completed);
    atom_setlong(&stats[1], x->frames_skipped);
    outlet_anything(x->info_outlet, gensym("stats"), 2, stats);
}

/* -----------------------------------------------------------------------
 * Task submission: called from perform routine (audio thread).
 * Computes the spectrum of input_frame, stores result in args, then
 * submits to the Max SDK thread pool.
 * ----------------------------------------------------------------------- */
void concat_task_submit(t_concat *x, const float *input_frame)
{
    t_concat_task_args *args = (t_concat_task_args*)
        sysmem_newptrclear(sizeof(t_concat_task_args));
    if (!args) {
        atom_setlong(&x->processing_flag, 0);
        return;
    }

    /* Spectrum buffer — freed in concat_task_complete */
    args->input_features = (float*)sysmem_newptr(x->n_bins * sizeof(float));
    if (!args->input_features) {
        sysmem_freeptr(args);
        atom_setlong(&x->processing_flag, 0);
        return;
    }

    args->owner        = x;
    args->frame_number = x->frames_completed + x->frames_skipped;
    args->timestamp    = systimer_gettime();

    /* Compute spectrum (audio thread can safely read feature_computer) */
    features_compute(x->feature_computer, input_frame, args->input_features);

    /* Submit to Max thread pool */
    threadpooltask_execute(
        (t_object*)x,
        (void*)args,
        (method)concat_task_worker,
        (method)concat_task_complete,
        &x->current_task,
        0
    );
}
