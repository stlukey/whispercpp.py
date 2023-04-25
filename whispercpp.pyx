#!python
# cython: language_level=3

import ffmpeg
import numpy as np
import requests
import os
from pathlib import Path

MODELS_DIR = str(Path('~/.cache/ggml-models').expanduser())
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

def sampling_strategy_from_string(strategy_string):
    strategy_map = {
        'GREEDY': whisper_sampling_strategy.WHISPER_SAMPLING_BEAM_SEARCH,
        'BEAM_SEARCH': whisper_sampling_strategy.WHISPER_SAMPLING_BEAM_SEARCH
    }
    return strategy_map[strategy_string.upper()]

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
    print("Sampling rate:", sr)
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

cdef whisper_full_params default_params(strategy='GREEDY'):
    strategy_value = sampling_strategy_from_string(strategy)
    cdef whisper_full_params params = whisper_full_default_params(
        strategy_value
    )
    params.print_realtime = True
    params.print_progress = True
    params.translate = False
    params.language = <const char *> LANGUAGE
    n_threads = N_THREADS
    return params


cdef class Whisper:
    """
    This class provides an interface for speech recognition using the Whisper library.

    Parameters:
    -----------
    model (str): Model to use for transcription. One of ['ggml-tiny', 'ggml-tiny.en', 'ggml-base',
            'ggml-base.en', 'ggml-small', 'ggml-small.en', 'ggml-medium', 'ggml-medium.en', 'ggml-large',
            'ggml-large-v1']. Defaults to 'ggml-base'.
    **kwargs: optional
        Additional arguments to override the default parameters for speech recognition. Accepts the following arguments:
            - strategy (str): Sampling strategy to use. Choose from 'GREEDY' or 'BEAM_SEARCH'. Default: 'GREEDY'.
            - print_progress (bool): Whether to print progress messages during transcription. Default: True.
            - print_realtime (bool): Whether to print transcription results in real time. Default: True.

    Attributes:
    -----------
    ctx: whisper_context *
        The pointer to the Whisper context used for speech recognition.
    params: whisper_full_params
        The parameters used for speech recognition.
    """
    cdef whisper_context * ctx
    cdef whisper_full_params params

    def __init__(self, model='tiny', **kwargs):
        model_fullname = f'ggml-{model}.bin'
        download_model(model_fullname)
        model_path = Path(MODELS_DIR).joinpath(model_fullname)
        cdef bytes model_b = str(model_path).encode('utf8')
        self.ctx = whisper_init_from_file(model_b)
        self.params = default_params(kwargs.get('strategy', 'GREEDY'))
        whisper_print_system_info()
        # Override default params
        self.params.print_progress = kwargs.get('print_progress', True)
        self.params.print_realtime = kwargs.get('print_realtime', True)


    def __dealloc__(self):
        whisper_free(self.ctx)

    def transcribe(self, filename=TEST_FILE, language=None):
        print("Transcribing...")
        cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames = load_audio(<bytes>filename, SAMPLE_RATE)
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


