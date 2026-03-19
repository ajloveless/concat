#include "ext.h"
#include "ext_obex.h"
#include "ext_path.h"
#include "ext_buffer.h"
#include "ext_strings.h"
#include "corpus.h"
#include "audio_features.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * corpus.c
 * ----------------------------------------------------------------------- */

long corpus_frame_count(long n_samples, long win)
{
    long hop = win / 2;
    if (n_samples < win) return 0;
    return (n_samples - win) / hop + 1;
}

t_corpus* corpus_create(long win, float alpha)
{
    t_corpus *c = (t_corpus*)calloc(1, sizeof(t_corpus));
    if (!c) return NULL;
    c->win    = win;
    c->hop    = win / 2;
    c->n_bins = win / 2 + 1;
    c->alpha  = alpha;
    return c;
}

void corpus_free(t_corpus *c)
{
    if (!c) return;
    free(c->features);
    free(c->audio);
    free(c);
}

/* -----------------------------------------------------------------------
 * Load from an in-memory sample buffer.
 * ----------------------------------------------------------------------- */
int corpus_load_samples(t_corpus *c, const float *samples, long n_samples)
{
    if (!c || !samples || n_samples < c->win) return -1;

    long n_frames = corpus_frame_count(n_samples, c->win);
    if (n_frames == 0) return -1;

    /* (Re-)allocate storage */
    free(c->features);
    free(c->audio);
    c->features = NULL;
    c->audio    = NULL;

    c->features = (float*)malloc(n_frames * c->n_bins * sizeof(float));
    c->audio    = (float*)malloc(n_frames * c->win  * sizeof(float));
    if (!c->features || !c->audio) {
        free(c->features); c->features = NULL;
        free(c->audio);    c->audio    = NULL;
        return -1;
    }

    /* Build feature computer */
    t_feature_computer *fc = features_create(c->win);
    if (!fc) { free(c->features); free(c->audio); c->features=NULL; c->audio=NULL; return -1; }

    long hop = c->hop;
    for (long k = 0; k < n_frames; k++) {
        const float *frame_in = samples + k * hop;

        /* Store windowed audio */
        float *frame_audio = c->audio + k * c->win;
        memcpy(frame_audio, frame_in, c->win * sizeof(float));
        utils_hann_window(frame_audio, c->win);

        /* Compute dB magnitude spectrum */
        float *frame_feat = c->features + k * c->n_bins;
        features_compute_db(fc, frame_in, frame_feat);
    }

    features_free(fc);
    c->n_frames = n_frames;
    return 0;
}

/* -----------------------------------------------------------------------
 * Load from raw float32 binary file.
 * ----------------------------------------------------------------------- */
int corpus_load_file(t_corpus *c, const char *path, double samplerate)
{
    (void)samplerate;  /* reserved for future use (e.g. resampling) */
    if (!c || !path) return -1;

    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    rewind(fp);

    long n_samples = file_size / sizeof(float);
    if (n_samples < c->win) { fclose(fp); return -1; }

    float *samples = (float*)malloc(n_samples * sizeof(float));
    if (!samples) { fclose(fp); return -1; }

    size_t read = fread(samples, sizeof(float), (size_t)n_samples, fp);
    fclose(fp);
    if ((long)read != n_samples) { free(samples); return -1; }

    int result = corpus_load_samples(c, samples, n_samples);
    free(samples);

    if (result == 0) {
        strncpy(c->path, path, sizeof(c->path) - 1);
        c->path[sizeof(c->path) - 1] = '\0';
    }
    return result;
}

/* -----------------------------------------------------------------------
 * Load from an audio file (AIFF / WAV / MP3) using Max's path system.
 * Must be called from the Max main thread (e.g. inside a defer'd callback).
 * ----------------------------------------------------------------------- */
int corpus_load_audio(t_corpus *c, t_object *owner, const char *filename)
{
    if (!c || !filename) return -1;

    char conformedname[MAX_PATH_CHARS];
    char fullpath[MAX_PATH_CHARS];
    short path_id;
    t_fourcc outtype;
    t_fourcc filetypes[3] = {'AIFF', 'WAVE', 'Mp3 '};

    /* Copy and conform the filename for Max path lookup */
    strncpy_zero(conformedname, filename, MAX_PATH_CHARS);

    /* Search Max's file paths for the file */
    if (locatefile_extended(conformedname, &path_id, &outtype, filetypes, 3)) {
        if (owner) object_error(owner, "concat~: cannot find file '%s'", filename);
        return -1;
    }

    /* Convert to an absolute system path */
    if (path_toabsolutesystempath(path_id, conformedname, fullpath)) {
        if (owner) object_error(owner, "concat~: cannot resolve path for '%s'", filename);
        return -1;
    }

    /* Create a temporary named buffer~ to decode the audio file.
     * Use the corpus pointer address to ensure the name is unique per
     * load operation (avoids conflicts with concurrent loads). */
    char buf_name_str[32];
    snprintf(buf_name_str, sizeof(buf_name_str), "_concat_tmp_%p", (void*)c);
    t_symbol *buf_name = gensym(buf_name_str);
    t_buffer_obj *buffer = buffer_new(buf_name, 0);
    if (!buffer) {
        if (owner) object_error(owner, "concat~: failed to create buffer");
        return -1;
    }

    /* Read the file into the buffer */
    t_atom path_atom;
    atom_setsym(&path_atom, gensym(fullpath));
    if (buffer_read(buffer, gensym(fullpath), 1, &path_atom)) {
        if (owner) object_error(owner, "concat~: failed to read '%s'", filename);
        buffer_free(buffer);
        return -1;
    }

    /* Lock buffer samples for reading */
    float *tab = buffer_locksamples(buffer);
    if (!tab) {
        if (owner) object_error(owner, "concat~: failed to lock buffer samples");
        buffer_free(buffer);
        return -1;
    }

    t_atom_long framecount   = buffer_getframecount(buffer);
    t_atom_long channelcount = buffer_getchannelcount(buffer);

    if (framecount == 0) {
        if (owner) object_error(owner, "concat~: empty file '%s'", filename);
        buffer_unlocksamples(buffer);
        buffer_free(buffer);
        return -1;
    }

    /* Mix down to mono: take the left channel when stereo */
    float *mono_samples = NULL;
    int need_free = 0;

    if (channelcount == 1) {
        mono_samples = tab;
    } else {
        mono_samples = (float*)malloc((size_t)framecount * sizeof(float));
        if (!mono_samples) {
            buffer_unlocksamples(buffer);
            buffer_free(buffer);
            return -1;
        }
        need_free = 1;
        for (long i = 0; i < framecount; i++) {
            mono_samples[i] = tab[i * channelcount]; /* left channel */
        }
    }

    /* Load samples into the corpus */
    int result = corpus_load_samples(c, mono_samples, (long)framecount);

    /* Cleanup */
    if (need_free) free(mono_samples);
    buffer_unlocksamples(buffer);
    buffer_free(buffer);

    if (result == 0 && owner) {
        object_post(owner, "concat~: loaded '%s' (%ld frames, %ld corpus windows)",
                    filename, (long)framecount, c->n_frames);
        strncpy(c->path, filename, sizeof(c->path) - 1);
        c->path[sizeof(c->path) - 1] = '\0';
    }

    return result;
}
