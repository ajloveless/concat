/* -----------------------------------------------------------------------
 * concat_tilde.c  -  Max/MSP external: concat~
 *
 * Real-time concatenative synthesis using a particle filter.
 * Background processing via the Max thread pool ensures the perform
 * routine never blocks.
 *
 * Attributes:
 *   @winsize     <int>    STFT window size       (default 2048)
 *   @particles   <int>    Number of particles    (default 1000)
 *   @sparsity    <int>    Activations/particle   (default 5)
 *   @pd          <float>  Temporal continuity    (default 0.95)
 *   @temperature <float>  Observation sharpness  (default 50.0)
 *   @wet         <float>  Wet/dry mix            (default 1.0)
 *   @nmfiters    <int>    NMF iterations         (default 30)
 *
 * Messages:
 *   corpus <path>  — load corpus (raw float32 file)
 *   reset          — reset particle filter
 *   stats          — post performance statistics
 *
 * Inlets:
 *   0: audio signal (target)
 *   1: messages
 *
 * Outlets:
 *   0: audio signal (synthesised)
 *   1: info / stats
 * ----------------------------------------------------------------------- */

#include "concat_tilde.h"
#include "concat_task.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * Class pointer (file-scope)
 * ----------------------------------------------------------------------- */
static t_class *concat_class = NULL;

/* -----------------------------------------------------------------------
 * Forward declarations
 * ----------------------------------------------------------------------- */
void *concat_new(t_symbol *s, long argc, t_atom *argv);
void  concat_free(t_concat *x);
void  concat_dsp64(t_concat *x, t_object *dsp64, short *count,
                   double samplerate, long maxvectorsize, long flags);
void  concat_perform64(t_concat *x, t_object *dsp64,
                       double **ins, long numins,
                       double **outs, long numouts,
                       long sampleframes, long flags, void *userparam);
void  concat_corpus(t_concat *x, t_symbol *s);
void  concat_reset(t_concat *x);
void  concat_stats(t_concat *x);
void  concat_assist(t_concat *x, void *b, long m, long a, char *s);

/* -----------------------------------------------------------------------
 * ext_main  — called once when the external is loaded into Max
 * ----------------------------------------------------------------------- */
void ext_main(void *r)
{
    /* Initialise the Max thread pool (idempotent global call) */
    threadpooltask_init();

    t_class *c = class_new("concat~",
                            (method)concat_new,
                            (method)concat_free,
                            sizeof(t_concat),
                            NULL, A_GIMME, 0);

    /* DSP */
    class_addmethod(c, (method)concat_dsp64,  "dsp64",  A_CANT, 0);
    class_addmethod(c, (method)concat_assist, "assist", A_CANT, 0);

    /* Messages */
    class_addmethod(c, (method)concat_corpus, "corpus", A_SYM,      0);
    class_addmethod(c, (method)concat_reset,  "reset",  A_NOTHING,  0);
    class_addmethod(c, (method)concat_stats,  "stats",  A_NOTHING,  0);

    /* Attributes */
    CLASS_ATTR_LONG(c,  "winsize",     0, t_concat, win);
    CLASS_ATTR_FILTER_MIN(c, "winsize", 64);
    CLASS_ATTR_LABEL(c, "winsize", 0, "Window Size");
    CLASS_ATTR_SAVE(c, "winsize", 0);

    CLASS_ATTR_LONG(c,  "particles",   0, t_concat, N_particles);
    CLASS_ATTR_FILTER_MIN(c, "particles", 1);
    CLASS_ATTR_LABEL(c, "particles", 0, "Number of Particles");
    CLASS_ATTR_SAVE(c, "particles", 0);

    CLASS_ATTR_LONG(c,  "sparsity",    0, t_concat, sparsity);
    CLASS_ATTR_FILTER_MIN(c, "sparsity", 1);
    CLASS_ATTR_LABEL(c, "sparsity", 0, "Activations per Particle");
    CLASS_ATTR_SAVE(c, "sparsity", 0);

    CLASS_ATTR_FLOAT(c, "pd",          0, t_concat, pd);
    CLASS_ATTR_LABEL(c, "pd", 0, "Temporal Continuity Probability");
    CLASS_ATTR_SAVE(c, "pd", 0);

    CLASS_ATTR_FLOAT(c, "temperature", 0, t_concat, temperature);
    CLASS_ATTR_FILTER_MIN(c, "temperature", 0.001f);
    CLASS_ATTR_LABEL(c, "temperature", 0, "Observation Temperature");
    CLASS_ATTR_SAVE(c, "temperature", 0);

    CLASS_ATTR_FLOAT(c, "wet",         0, t_concat, wet);
    CLASS_ATTR_LABEL(c, "wet", 0, "Wet/Dry Mix");
    CLASS_ATTR_SAVE(c, "wet", 0);

    CLASS_ATTR_LONG(c,  "nmfiters",    0, t_concat, nmf_iters);
    CLASS_ATTR_FILTER_MIN(c, "nmfiters", 1);
    CLASS_ATTR_LABEL(c, "nmfiters", 0, "NMF Iterations");
    CLASS_ATTR_SAVE(c, "nmfiters", 0);

    class_dspinit(c);
    class_register(CLASS_BOX, c);
    concat_class = c;
}

/* -----------------------------------------------------------------------
 * Object constructor
 * ----------------------------------------------------------------------- */
void* concat_new(t_symbol *s, long argc, t_atom *argv)
{
    t_concat *x = (t_concat*)object_alloc(concat_class);
    if (!x) return NULL;

    /* Defaults */
    x->win         = 2048;
    x->N_particles = 1000;
    x->sparsity    = 5;
    x->pd          = 0.95f;
    x->temperature = 50.0f;
    x->wet         = 1.0f;
    x->nmf_iters   = 30;

    /* Process attribute arguments (e.g. [concat~ @winsize 1024]) */
    attr_args_process(x, argc, argv);

    /* Derived values */
    x->hop    = x->win / 2;
    x->n_bins = x->win / 2 + 1;

    /* DSP setup: 1 signal inlet.
     * In Max, outlets are numbered left-to-right (0 = leftmost) but
     * outlet_new() appends outlets from right to left.  Therefore the
     * FIRST call to outlet_new creates the highest-numbered (rightmost)
     * outlet, and the LAST call creates outlet 0 (leftmost signal outlet).
     *   1st call → outlet 1 (rightmost): info/stats
     *   2nd call → outlet 0 (leftmost):  audio signal */
    dsp_setup((t_pxobject*)x, 1);
    x->info_outlet = outlet_new((t_object*)x, NULL);        /* outlet 1 (rightmost) */
    outlet_new((t_object*)x, "signal");                     /* outlet 0 (leftmost)  */

    /* Double-buffered output */
    x->output_bufs[0] = concat_buffer_create(x->win);
    x->output_bufs[1] = concat_buffer_create(x->win);
    if (!x->output_bufs[0] || !x->output_bufs[1]) {
        object_error((t_object*)x, "concat~: out of memory (output buffers)");
        concat_free(x);
        return NULL;
    }
    atom_setlong(&x->read_idx,  0);
    atom_setlong(&x->write_idx, 1);
    atom_setlong(&x->processing_flag, 0);

    /* Input frame accumulation */
    x->input_frame = (float*)sysmem_newptrclear(x->win * sizeof(float));
    if (!x->input_frame) {
        object_error((t_object*)x, "concat~: out of memory (input frame)");
        concat_free(x);
        return NULL;
    }
    x->input_frame_pos = 0;

    /* Feature computer */
    x->feature_computer = features_create(x->win);
    if (!x->feature_computer) {
        object_error((t_object*)x, "concat~: out of memory (feature computer)");
        concat_free(x);
        return NULL;
    }

    /* Empty corpus (populated by 'corpus' message) */
    x->corpus = corpus_create(x->win, 1.0f);
    if (!x->corpus) {
        object_error((t_object*)x, "concat~: out of memory (corpus)");
        concat_free(x);
        return NULL;
    }
    x->corpus_loaded = 0;

    /* Particle filter (placeholder corpus size of 1 until corpus loaded) */
    x->pf = particle_filter_create(x->N_particles,
                                    x->sparsity,
                                    1,
                                    x->pd,
                                    x->temperature,
                                    x->nmf_iters);
    if (!x->pf) {
        object_error((t_object*)x, "concat~: out of memory (particle filter)");
        concat_free(x);
        return NULL;
    }

    /* Mutex protecting corpus + particle filter during reload */
    systhread_mutex_new(&x->corpus_mutex, SYSTHREAD_MUTEX_NORMAL);

    x->frames_skipped   = 0;
    x->frames_completed = 0;
    x->current_task     = NULL;

    return x;
}

/* -----------------------------------------------------------------------
 * Object destructor
 * ----------------------------------------------------------------------- */
void concat_free(t_concat *x)
{
    /* Cancel any in-flight task and wait for all tasks to finish */
    if (x->current_task) {
        threadpooltask_cancel(x->current_task);
        x->current_task = NULL;
    }
    threadpooltask_join_object((t_object*)x);

    dsp_free((t_pxobject*)x);

    particle_filter_free(x->pf);           x->pf = NULL;
    corpus_free(x->corpus);                x->corpus = NULL;
    features_free(x->feature_computer);   x->feature_computer = NULL;

    concat_buffer_free(x->output_bufs[0]); x->output_bufs[0] = NULL;
    concat_buffer_free(x->output_bufs[1]); x->output_bufs[1] = NULL;

    sysmem_freeptr(x->input_frame);        x->input_frame = NULL;

    systhread_mutex_free(x->corpus_mutex);
}

/* -----------------------------------------------------------------------
 * DSP setup
 * ----------------------------------------------------------------------- */
void concat_dsp64(t_concat *x, t_object *dsp64, short *count,
                  double samplerate, long maxvectorsize, long flags)
{
    /* Cancel running task so buffers can be safely reset */
    if (x->current_task) {
        threadpooltask_cancel(x->current_task);
        x->current_task = NULL;
    }

    concat_buffer_reset(x->output_bufs[0]);
    concat_buffer_reset(x->output_bufs[1]);
    atom_setlong(&x->processing_flag, 0);
    atom_setlong(&x->read_idx,  0);
    atom_setlong(&x->write_idx, 1);
    x->input_frame_pos = 0;
    if (x->input_frame) memset(x->input_frame, 0, (size_t)x->win * sizeof(float));

    object_method(dsp64, gensym("dsp_add64"), x, concat_perform64, 0, NULL);
}

/* -----------------------------------------------------------------------
 * Perform routine  (audio thread — MUST NOT block)
 * ----------------------------------------------------------------------- */
void concat_perform64(t_concat *x, t_object *dsp64,
                      double **ins, long numins,
                      double **outs, long numouts,
                      long sampleframes, long flags, void *userparam)
{
    double *in  = ins[0];
    double *out = outs[0];
    long    n   = sampleframes;
    long    hop = x->hop;
    float   wet = x->wet;

    /* Snapshot read buffer index for this vector */
    long read_idx = atom_getlong(&x->read_idx);
    t_concat_output_buffer *read_buf = x->output_bufs[read_idx];

    while (n--) {
        /* Accumulate input sample */
        x->input_frame[x->input_frame_pos++] = (float)(*in++);

        /* Output: read from ready buffer, overlap-add, or silence */
        float out_sample = 0.0f;
        if (atom_getlong(&read_buf->ready) && read_buf->hop_offset < hop) {
            out_sample = read_buf->audio[read_buf->hop_offset++];
        }
        *out++ = (double)(wet * out_sample);

        /* Full window accumulated → submit to thread pool */
        if (x->input_frame_pos >= x->win) {
            if (atom_getlong(&x->processing_flag) == 0) {
                atom_setlong(&x->processing_flag, 1);
                concat_task_submit(x, x->input_frame);
            } else {
                /* Worker still busy: skip this frame (graceful degradation) */
                x->frames_skipped++;
            }
            x->input_frame_pos = 0;
        }
    }
}

/* -----------------------------------------------------------------------
 * Corpus loading  (background thread)
 * ----------------------------------------------------------------------- */
typedef struct _corpus_load_args {
    t_concat *x;
    char      path[512];
} t_corpus_load_args;

static void* corpus_load_thread(void *arg)
{
    t_corpus_load_args *la = (t_corpus_load_args*)arg;
    t_concat *x = la->x;

    /* The mutex protects both the corpus pointer and the particle filter.
     * The worker thread acquires the same mutex before reading the corpus,
     * so holding it here prevents any race during the swap. */
    systhread_mutex_lock(x->corpus_mutex);

    t_corpus *nc = corpus_create(x->win, 1.0f);
    if (nc && corpus_load_file(nc, la->path, 44100.0) == 0) {
        corpus_free(x->corpus);
        x->corpus = nc;
        particle_filter_set_corpus_size(x->pf, nc->n_frames);
        x->corpus_loaded = 1;
        object_post((t_object*)x,
                    "concat~: loaded '%s' (%ld frames)", la->path, nc->n_frames);
    } else {
        corpus_free(nc);
        object_error((t_object*)x, "concat~: failed to load '%s'", la->path);
    }

    systhread_mutex_unlock(x->corpus_mutex);

    sysmem_freeptr(la);
    return NULL;
}

void concat_corpus(t_concat *x, t_symbol *s)
{
    if (!s || s == gensym("")) {
        object_error((t_object*)x, "concat~: corpus <path>");
        return;
    }

    t_corpus_load_args *la = (t_corpus_load_args*)
        sysmem_newptrclear(sizeof(t_corpus_load_args));
    if (!la) { object_error((t_object*)x, "concat~: out of memory"); return; }
    la->x = x;
    strncpy(la->path, s->s_name, sizeof(la->path) - 1);

    t_systhread thr;
    systhread_create((method)corpus_load_thread, la, 0, 0, 0, &thr);
}

/* -----------------------------------------------------------------------
 * Message handlers
 * ----------------------------------------------------------------------- */
void concat_reset(t_concat *x)
{
    systhread_mutex_lock(x->corpus_mutex);
    particle_filter_reset(x->pf);
    concat_buffer_reset(x->output_bufs[0]);
    concat_buffer_reset(x->output_bufs[1]);
    x->input_frame_pos  = 0;
    x->frames_skipped   = 0;
    x->frames_completed = 0;
    atom_setlong(&x->processing_flag, 0);
    systhread_mutex_unlock(x->corpus_mutex);
    object_post((t_object*)x, "concat~: reset");
}

void concat_stats(t_concat *x)
{
    t_atom stats[2];
    atom_setlong(&stats[0], x->frames_completed);
    atom_setlong(&stats[1], x->frames_skipped);
    outlet_anything(x->info_outlet, gensym("stats"), 2, stats);
}

/* -----------------------------------------------------------------------
 * Assist strings
 * ----------------------------------------------------------------------- */
void concat_assist(t_concat *x, void *b, long m, long a, char *s)
{
    if (m == ASSIST_INLET) {
        switch (a) {
            case 0: sprintf(s, "(signal) Target audio input");         break;
            case 1: sprintf(s, "(msg) corpus <path> / reset / stats"); break;
            default: break;
        }
    } else {
        switch (a) {
            case 0: sprintf(s, "(signal) Synthesised output");                    break;
            case 1: sprintf(s, "(list) stats: frames_completed frames_skipped");  break;
            default: break;
        }
    }
}
