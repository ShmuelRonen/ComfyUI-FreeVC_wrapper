from scipy.ndimage.morphology import binary_dilation
from speaker_encoder.params_data import *
from pathlib import Path
from typing import Optional, Union
import numpy as np
import librosa
import struct

int16_max = (2 ** 15) - 1

def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    
    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)
    
    return wav

def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        y=wav,
        sr=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T

def trim_long_silences(wav):
    """
    Alternative implementation using librosa instead of webrtcvad.
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.
    """
    # Convert to float32 for librosa
    wav_float = wav.astype(np.float32)
    
    # Calculate frame length and hop length based on VAD parameters
    frame_length = int(vad_window_length * sampling_rate / 1000)
    hop_length = frame_length // 2
    
    # Get non-silent intervals using librosa
    intervals = librosa.effects.split(
        wav_float,
        top_db=30,  # Adjust this value to match webrtcvad sensitivity
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    # Handle edge case where no speech is detected
    if len(intervals) == 0:
        return wav
        
    # Merge intervals that are close together
    merged_intervals = []
    current_start, current_end = intervals[0]
    max_silence_length = vad_max_silence_length * frame_length
    
    for start, end in intervals[1:]:
        if start - current_end <= max_silence_length:
            current_end = end
        else:
            merged_intervals.append((current_start, current_end))
            current_start, current_end = start, end
    merged_intervals.append((current_start, current_end))
    
    # Concatenate all voiced segments
    wav_output = []
    for start, end in merged_intervals:
        wav_output.extend(wav[start:end])
    
    return np.array(wav_output)

def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean(wav ** 2))
    wave_dBFS = 20 * np.log10(rms)
    dBFS_change = target_dBFS - wave_dBFS
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))