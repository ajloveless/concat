/* -----------------------------------------------------------------------
 * threadpooltask.c  -  Thread pool task implementation for Max/MSP
 *
 * Provides a lightweight background task system built on the Max SDK's
 * systhread API.  Each submitted task is executed in a dedicated
 * background thread.  An owner registry allows callers to wait for all
 * tasks belonging to a particular object to complete via
 * threadpooltask_join_object().
 *
 * Thread safety model:
 *   - g_reg_mutex protects the owner registry and g_next_id counter.
 *   - The task struct's `state` field is written by the runner thread
 *     (QUEUED -> RUNNING -> DONE) and may also be set to CANCELLED by
 *     the caller via threadpooltask_cancel().  The cancel write races
 *     with the runner's state reads; the result is a benign TOCTOU (at
 *     worst, the completion callback fires after cancellation was
 *     requested).
 *   - Task struct lifetime:
 *       * Allocated by threadpooltask_execute(), stored at *taskptr.
 *       * The runner does NOT free the struct; ownership stays with the
 *         caller.  This avoids use-after-free when the caller still
 *         holds x->current_task after the runner finishes.
 *       * The caller is responsible for freeing the struct.  The
 *         recommended pattern is to free it inside the main-thread
 *         completion handler (e.g. concat_task_finish) via
 *         sysmem_freeptr(x->current_task) before clearing the pointer.
 *       * threadpooltask_join_object() returns once reg_dec() has been
 *         called by the runner, guaranteeing the runner has finished
 *         with the struct.  After join_object returns, a cancel+NULL
 *         pattern leaves a small one-off leak; this is acceptable since
 *         it only occurs during infrequent DSP recompilation or object
 *         destruction.
 * ----------------------------------------------------------------------- */

#include "threadpooltask.h"

#include "ext.h"
#include "ext_obex.h"
#include "ext_systhread.h"

#include <string.h>

/* -----------------------------------------------------------------------
 * Task states
 * ----------------------------------------------------------------------- */
#define TSTATE_QUEUED    1
#define TSTATE_RUNNING   2
#define TSTATE_DONE      4
#define TSTATE_CANCELLED 8

/* -----------------------------------------------------------------------
 * Extended task struct — the public t_threadpooltask MUST be the first
 * member so that a pointer to the public part equals a pointer to the
 * extended struct (used in threadpooltask_join).
 * ----------------------------------------------------------------------- */
typedef struct _task_ext {
    t_threadpooltask pub;    /* public — MUST be first */
    t_systhread      thread; /* handle used by threadpooltask_join() */
} t_task_ext;

/* -----------------------------------------------------------------------
 * Owner registry — tracks how many tasks are still in flight per owner
 * ----------------------------------------------------------------------- */
#define MAX_OWNERS 64

typedef struct _owner_rec {
    t_object *owner;
    long      pending;
} t_owner_rec;

static t_owner_rec       g_owners[MAX_OWNERS];
static t_systhread_mutex g_reg_mutex;
static long              g_next_id     = 0;
static int               g_initialized = 0;

/* ---- Registry helpers ---- */

static void reg_add(t_object *owner)
{
    int i;
    systhread_mutex_lock(g_reg_mutex);
    for (i = 0; i < MAX_OWNERS; i++) {
        if (g_owners[i].owner == owner) {
            g_owners[i].pending++;
            systhread_mutex_unlock(g_reg_mutex);
            return;
        }
    }
    for (i = 0; i < MAX_OWNERS; i++) {
        if (!g_owners[i].owner) {
            g_owners[i].owner   = owner;
            g_owners[i].pending = 1;
            systhread_mutex_unlock(g_reg_mutex);
            return;
        }
    }
    /* Registry full — log and continue (task pending count won't be tracked) */
    object_error(owner, "threadpooltask: owner registry full (max %d owners)",
                 MAX_OWNERS);
    systhread_mutex_unlock(g_reg_mutex);
}

static void reg_dec(t_object *owner)
{
    int i;
    systhread_mutex_lock(g_reg_mutex);
    for (i = 0; i < MAX_OWNERS; i++) {
        if (g_owners[i].owner == owner) {
            if (--g_owners[i].pending <= 0) {
                g_owners[i].owner   = NULL;
                g_owners[i].pending = 0;
            }
            break;
        }
    }
    systhread_mutex_unlock(g_reg_mutex);
}

static long reg_pending(t_object *owner)
{
    int  i;
    long n = 0;
    systhread_mutex_lock(g_reg_mutex);
    for (i = 0; i < MAX_OWNERS; i++) {
        if (g_owners[i].owner == owner) {
            n = g_owners[i].pending;
            break;
        }
    }
    systhread_mutex_unlock(g_reg_mutex);
    return n;
}

/* -----------------------------------------------------------------------
 * Task runner — executes in a background thread.
 * NOTE: does NOT free the task struct.  The caller (main-thread
 * completion handler) owns the struct and must free it.
 * ----------------------------------------------------------------------- */
static void *task_runner(void *arg)
{
    t_task_ext       *te   = (t_task_ext *)arg;
    t_threadpooltask *task = &te->pub;
    t_object         *owner = task->owner;

    /* Skip work if cancelled before we even started */
    if (task->state != TSTATE_CANCELLED) {
        task->state = TSTATE_RUNNING;

        /* Worker callback (heavy computation) */
        if (task->cbtask)
            ((void (*)(void *))task->cbtask)(task->args);

        /* Completion callback — only if not cancelled during worker */
        if (task->state != TSTATE_CANCELLED && task->cbcomplete)
            ((void (*)(void *))task->cbcomplete)(task->args);
    }

    /* Mark done and decrement owner counter.
     * threadpooltask_join_object() polls until pending reaches zero,
     * so reg_dec() must be the last thing the runner does. */
    task->state = TSTATE_DONE;
    reg_dec(owner);

    return NULL;
}

/* -----------------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------------- */

void threadpooltask_init(void)
{
    if (g_initialized) return;
    memset(g_owners, 0, sizeof(g_owners));
    systhread_mutex_new(&g_reg_mutex, SYSTHREAD_MUTEX_NORMAL);
    g_initialized = 1;
}

long threadpooltask_execute(t_object *owner, void *args,
                             method cbtask, method cbcomplete,
                             t_threadpooltask **taskptr, long flags)
{
    t_task_ext *te;
    long        ret;
    long        new_id;

    if (!g_initialized) threadpooltask_init();

    te = (t_task_ext *)sysmem_newptrclear(sizeof(t_task_ext));
    if (!te) return 1;

    /* Atomically claim a task ID */
    systhread_mutex_lock(g_reg_mutex);
    new_id = ++g_next_id;
    systhread_mutex_unlock(g_reg_mutex);

    te->pub.flags      = flags;
    te->pub.state      = TSTATE_QUEUED;
    te->pub.id         = new_id;
    te->pub.owner      = owner;
    te->pub.args       = args;
    te->pub.cbtask     = cbtask;
    te->pub.cbcomplete = cbcomplete;

    /* Register the pending task for this owner */
    reg_add(owner);

    if (taskptr) *taskptr = &te->pub;

    ret = systhread_create((method)task_runner, te, 0, 0, 0, &te->thread);
    if (ret != 0) {
        reg_dec(owner);
        sysmem_freeptr(te);
        if (taskptr) *taskptr = NULL;
        return 1;
    }

    return 0;
}

long threadpooltask_execute_method(t_object *obtask, t_symbol *mtask,
                                   long actask, t_atom *avtask,
                                   t_object *obcomp, t_symbol *mcomp,
                                   long accomp, t_atom *avcomp,
                                   t_threadpooltask **task, long flags)
{
    /* Not required for concat~ — use threadpooltask_execute() instead */
    return 1;
}

void threadpooltask_purge_object(t_object *owner)
{
    /* No central task list is maintained, so tasks cannot be enumerated
     * by owner here.  Use threadpooltask_cancel() on individual task
     * handles before calling threadpooltask_join_object(). */
}

void threadpooltask_join_object(t_object *owner)
{
    /* Poll the registry until no tasks remain in flight for this owner.
     * The runner calls reg_dec() as its final act, so once pending
     * reaches zero the runner has fully finished with the task struct. */
    while (reg_pending(owner) > 0)
        systhread_sleep(1); /* 1 ms poll interval */
}

long threadpooltask_cancel(t_threadpooltask *task)
{
    if (!task) return 1;
    task->state = TSTATE_CANCELLED;
    return 0;
}

long threadpooltask_join(t_threadpooltask *task)
{
    void *retval = NULL;
    if (!task) return 1;
    /* The public t_threadpooltask is the first member of t_task_ext,
     * so the cast is valid. */
    systhread_join(((t_task_ext *)task)->thread, &retval);
    return 0;
}
