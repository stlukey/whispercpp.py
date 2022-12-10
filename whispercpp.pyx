#!python
# cython: language_level=3
# distutils: language = c++
# distutils: sources= ./whisper.cpp/whisper.cpp ./whisper.cpp/ggml.c

import ffmpeg
import numpy as np
cimport numpy as cnp

cdef int SAMPLE_RATE = 16000
cdef char* TEST_FILE = b'test.wav'
cdef char* DEFAULT_MODEL = b'ggml-tiny.bin'
cdef char* LANGUAGE = b'fr'

cdef audio_data load_audio(bytes file, int sr = SAMPLE_RATE):
    out = (
        ffmpeg.input(file, threads=0)
        .output(
            "-", format="s16le",
            acodec="pcm_s16le",
            ac=1, ar=sr
        )
        .run(
            cmd=["ffmpeg", "-nostdin"],
            capture_stdout=True,
            capture_stderr=True
        )
    )[0]

    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames = (
        np.frombuffer(out, np.int16)
        .flatten()
        .astype(np.float32)
    ) / pow(2, 15)

    cdef audio_data data;
    data.frames = &frames[0]
    data.n_frames = len(frames)

    return data

cdef whisper_full_params default_params():
    cdef whisper_full_params params = whisper_full_default_params(
        whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
    )
    params.print_realtime = True
    params.print_progress = True
    params.translate = False
    params.language = <const char *> LANGUAGE
    return params


cdef class Whisper:
    cdef whisper_context * ctx
    cdef whisper_full_params params

    def __init__(self, char* model=DEFAULT_MODEL):
        self.ctx = whisper_init(model)
        self.params = default_params()

    def __dealloc__(self):
        whisper_free(self.ctx)

    cpdef str transcribe(self):
        cdef audio_data data = load_audio(TEST_FILE) 
        cdef int res = whisper_full(self.ctx, self.params, data.frames, data.n_frames)
        if res != 0:
            raise RuntimeError
        cdef int n_segments = whisper_full_n_segments(self.ctx)
        return b'\n'.join([
            whisper_full_get_segment_text(self.ctx, i) for i in range(n_segments)
        ]).decode()




