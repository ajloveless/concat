#ifndef CONCAT_TASK_H
#define CONCAT_TASK_H

/* -----------------------------------------------------------------------
 * concat_task.h  -  Thread pool wrapper and task management
 *
 * Call sequence:
 *   perform  → concat_task_submit()      [non-blocking, audio thread]
 *   worker   → concat_task_worker()      [heavy computation, background]
 *   callback → concat_task_complete()    [background, defers to main]
 *   main     → concat_task_finish()      [buffer swap + stats, main thread]
 * ----------------------------------------------------------------------- */

#include "ext.h"
#include "ext_obex.h"

/* Forward-declare t_concat so this header is self-contained. */
struct _concat;
typedef struct _concat t_concat;

typedef struct _concat_task_args {
    t_concat *owner;
    float    *input_features;  /* n_bins spectrum (allocated in submit, freed in complete) */
    long      frame_number;
    double    timestamp;
} t_concat_task_args;

/* Submit a particle filter task to the Max thread pool (non-blocking). */
void concat_task_submit(t_concat *x, const float *input_frame);

/* Worker: heavy computation in background thread. */
void concat_task_worker(t_concat_task_args *args);

/* Completion: still in background thread; defers to main via defer_low. */
void concat_task_complete(t_concat_task_args *args);

/* Main-thread callback: buffer swap and outlet output. */
void concat_task_finish(t_concat *x, t_symbol *sym, long argc, void *argv);

#endif /* CONCAT_TASK_H */
