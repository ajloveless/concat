#ifndef CONCAT_BUFFER_H
#define CONCAT_BUFFER_H

/* -----------------------------------------------------------------------
 * concat_buffer.h  -  Double-buffer management with atomic operations
 *
 * Uses the Max SDK t_atom_long and atom_setlong/atom_getlong for
 * portable atomic access on the ready flag.
 * ----------------------------------------------------------------------- */

#include "ext.h"
#include "ext_obex.h"
#include "ext_atomic.h"

typedef struct _concat_output_buffer {
    float        *audio;        /* win synthesized samples */
    long          hop_offset;   /* read cursor (perform reads hop at a time) */
    t_atom_long   ready;        /* 0 = being written, 1 = ready to read */
    long          win;          /* window size */
} t_concat_output_buffer;

/* Allocate a buffer for win samples.  Returns NULL on failure. */
t_concat_output_buffer* concat_buffer_create(long win);

/* Free the buffer. */
void concat_buffer_free(t_concat_output_buffer *buf);

/* Zero the audio content and reset hop_offset and ready flag. */
void concat_buffer_reset(t_concat_output_buffer *buf);

#endif /* CONCAT_BUFFER_H */
