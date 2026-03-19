#ifndef CONCAT_TILDE_H
#define CONCAT_TILDE_H

/* -----------------------------------------------------------------------
 * concat_tilde.h  -  Shared type definitions for the concat~ external
 *
 * This header is included by both concat_tilde.c and concat_task.c so
 * that both compilation units share the same t_concat definition without
 * creating a circular dependency.
 * ----------------------------------------------------------------------- */

#include "ext.h"
#include "ext_obex.h"
#include "ext_atomic.h"
#include "z_dsp.h"
#include "ext_systhread.h"

#include "concat_buffer.h"
#include "particle.h"
#include "corpus.h"
#include "audio_features.h"

/* -----------------------------------------------------------------------
 * Main object structure
 * ----------------------------------------------------------------------- */
typedef struct _concat {
    t_pxobject obj;

    /* ---- Parameters ---- */
    long  win;
    long  hop;
    long  N_particles;
    long  sparsity;
    float pd;
    float temperature;
    float wet;
    long  nmf_iters;
    long  n_bins;

    /* ---- Double-buffered output ---- */
    t_concat_output_buffer *output_bufs[2];
    t_atom  read_idx;
    t_atom  write_idx;

    /* ---- Task management ---- */
    void        *current_task;       /* t_threadpooltask* (opaque) */
    t_atom  processing_flag;    /* 0=idle, 1=working */

    /* ---- Input frame accumulation ---- */
    float *input_frame;
    long   input_frame_pos;

    /* ---- Algorithm objects ---- */
    t_particle_filter  *pf;
    t_corpus           *corpus;
    t_feature_computer *feature_computer;

    /* ---- Corpus loading ---- */
    t_systhread_mutex  corpus_mutex;
    long               corpus_loaded;

    /* ---- Statistics ---- */
    long  frames_skipped;
    long  frames_completed;

    /* ---- Outlets ---- */
    void *info_outlet;
} t_concat;

/* Forward declarations of methods used by concat_task.c */
void concat_task_finish(t_concat *x, t_symbol *sym, long argc, void *argv);

#endif /* CONCAT_TILDE_H */
