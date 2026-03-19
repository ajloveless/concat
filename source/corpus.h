#ifndef CONCAT_CORPUS_H
#define CONCAT_CORPUS_H

/* -----------------------------------------------------------------------
 * corpus.h  -  Corpus loading and windowed feature/audio storage
 *
 * Mirrors audioutils.py::get_windowed which:
 *   1. Loads audio (single file or directory)
 *   2. Cuts into overlapping Hann-windowed frames (hop = win/2)
 *   3. Computes RFFT and extracts dB magnitude bins
 *   4. Stores both the spectral features AND the raw windowed samples
 * ----------------------------------------------------------------------- */

#include "audio_features.h"

typedef struct _corpus {
    long  win;          /* window size */
    long  hop;          /* hop size (win/2) */
    long  n_bins;       /* spectral bins per frame (win/2+1) */
    long  n_frames;     /* total number of corpus frames */

    /* features[k * n_bins + m] = dB magnitude of bin m in frame k
     * Column-major so that features + k*n_bins is the k-th frame vector. */
    float *features;    /* n_frames * n_bins */

    /* audio[k * win + i] = Hann-windowed sample i of frame k */
    float *audio;       /* n_frames * win */

    /* Per-frame regularisation weight (alpha) - uniform for now */
    float  alpha;

    /* Source info (for display / stats only) */
    char   path[512];
} t_corpus;

/* -----------------------------------------------------------------------
 * API
 * ----------------------------------------------------------------------- */

/* Allocate an empty corpus.  Returns NULL on failure. */
t_corpus* corpus_create(long win, float alpha);

/* Free all resources. */
void corpus_free(t_corpus *corpus);

/* Load corpus from a raw 32-bit float audio file.
 *   path      : absolute path to raw PCM float32 file
 *   samplerate: file sample rate (used only for metadata)
 *   win       : window size (must match corpus->win)
 *
 * Returns 0 on success, -1 on failure.
 *
 * NOTE: In a real Max external this would call sfopen / sndbuf APIs.
 * Here we read a flat binary float32 file so the implementation compiles
 * without the Max SDK; the concat_tilde.c integration layer provides a
 * thin wrapper that calls corpus_load_samples() with audio already decoded
 * by the Max buffer~ system.
 */
int corpus_load_file(t_corpus *corpus, const char *path, double samplerate);

/* Load corpus from an in-memory float buffer (already decoded audio).
 *   samples  : n_samples mono float32 samples
 *   n_samples: total number of samples
 * Returns 0 on success, -1 on failure. */
int corpus_load_samples(t_corpus *corpus, const float *samples, long n_samples);

/* Return the number of frames that would be produced for n_samples. */
long corpus_frame_count(long n_samples, long win);

#endif /* CONCAT_CORPUS_H */
