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
