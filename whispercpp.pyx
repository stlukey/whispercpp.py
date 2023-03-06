#!python
# cython: language_level=3

import ffmpeg
import numpy as np
import requests
import os
from pathlib import Path

MODELS_DIR = str(Path('~/.ggml-models').expanduser())
print("Saving models to:", MODELS_DIR)


cimport numpy as cnp
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef int SAMPLE_RATE = 16000
cdef char* TEST_FILE = 'test.wav'
cdef char* DEFAULT_MODEL = 'tiny'
cdef char* LANGUAGE = NULL
cdef int N_THREADS = os.cpu_count()

MODELS = {
    'ggml-tiny.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin',
    'ggml-tiny.en.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin',
    'ggml-base.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-base.bin',
    'ggml-base.en.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin',
    'ggml-small.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-small.bin',
    'ggml-small.en.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin',
    'ggml-medium.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin',
    'ggml-medium.en.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin',
    'ggml-large.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-large.bin',
    'ggml-large-v1.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin',
}

def model_exists(model):
    return os.path.exists(Path(MODELS_DIR).joinpath(model))

def download_model(model):
    if model_exists(model):
        return
    print(f'Downloading {model}...')

    url = MODELS[model]
    r = requests.get(url, allow_redirects=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(Path(MODELS_DIR).joinpath(model), 'wb') as f:
        f.write(r.content)


cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] load_audio(bytes file, int sr = SAMPLE_RATE):
    try:
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
    except FileNotFoundError:
        raise RuntimeError(f"File '{file}' not found")

    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames = (
        np.frombuffer(out, np.int16)
        .flatten()
        .astype(np.float32)
    ) / pow(2, 15)

    return frames

cdef whisper_full_params default_params() nogil:
    cdef whisper_full_params params = whisper_full_default_params(
        whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
    )
    params.print_realtime = True
    params.print_progress = True
    params.translate = False
    params.language = <const char *> LANGUAGE
    n_threads = N_THREADS
    return params


cdef class Whisper:
    cdef whisper_context * ctx
    cdef whisper_full_params params

    def __init__(self, model=DEFAULT_MODEL, pb=None):
        model_fullname = f'ggml-{model}.bin'
        download_model(model_fullname)
        model_path = Path(MODELS_DIR).joinpath(model_fullname)
        cdef bytes model_b = str(model_path).encode('utf8')
        self.ctx = whisper_init_from_file(model_b)
        self.params = default_params()
        whisper_print_system_info()

    def __dealloc__(self):
        whisper_free(self.ctx)

    def transcribe(self, filename=TEST_FILE, language=None):
        print("Transcribing...")
        cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames = load_audio(<bytes>filename)
        if language:
            print("Language:", language)
            LANGUAGE = language.encode('utf-8')
            self.params.language = LANGUAGE
        else:
            self.params.language = NULL
        transcript = whisper_full(self.ctx, self.params, &frames[0], len(frames))
        return transcript
    
    def extract_text(self, int res):
        print("Extracting text...")
        if res != 0:
            raise RuntimeError
        cdef int n_segments = whisper_full_n_segments(self.ctx)
        return [
            whisper_full_get_segment_text(self.ctx, i).decode() for i in range(n_segments)
        ]


