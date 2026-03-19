#include "concat_buffer.h"
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * concat_buffer.c
 * ----------------------------------------------------------------------- */

t_concat_output_buffer* concat_buffer_create(long win)
{
    t_concat_output_buffer *buf =
        (t_concat_output_buffer*)sysmem_newptrclear(sizeof(t_concat_output_buffer));
    if (!buf) return NULL;

    buf->audio = (float*)sysmem_newptrclear((long)(win * sizeof(float)));
    if (!buf->audio) { sysmem_freeptr(buf); return NULL; }

    buf->win        = win;
    buf->hop_offset = 0;
    atom_setlong(&buf->ready, 0);

    return buf;
}

void concat_buffer_free(t_concat_output_buffer *buf)
{
    if (!buf) return;
    sysmem_freeptr(buf->audio);
    sysmem_freeptr(buf);
}

void concat_buffer_reset(t_concat_output_buffer *buf)
{
    if (!buf) return;
    memset(buf->audio, 0, (size_t)buf->win * sizeof(float));
    buf->hop_offset = 0;
    atom_setlong(&buf->ready, 0);
}
