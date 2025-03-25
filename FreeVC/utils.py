import os
import sys
import json
import torch
import librosa
import numpy as np
from scipy.io.wavfile import write
from scipy import signal
from transformers import WavLMModel

# Try to import noisereduce, install if not available
try:
    import noisereduce as nr
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "noisereduce"])
    import noisereduce as nr

# Add FreeVC directory to path with higher priority to avoid conflicts
freevc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "freevc")
if freevc_dir not in sys.path:
    sys.path.insert(0, freevc_dir)  # Insert at beginning to prioritize

# Import FreeVC modules
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from speaker_encoder.voice_encoder import SpeakerEncoder

def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            data = f.read()
        config = json.loads(data)
        return HParams(**config)
    except Exception as e:
        print(f"Error loading config from {config_path}: {str(e)}")
        raise

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    assert os.path.isfile(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"
    try:
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        iteration = checkpoint_dict['iteration']
        saved_state_dict = checkpoint_dict['model']
        
        if hasattr(model, 'module'):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
            
        new_state_dict = {}
        for k, v in state_dict.items():
            try:
                new_state_dict[k] = saved_state_dict[k]
            except:
                print("%s is not in the checkpoint" % k)
                new_state_dict[k] = v
                
        if hasattr(model, 'module'):
            model.module.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(new_state_dict)
            
        return model, optimizer, iteration
    except Exception as e:
        print(f"Error loading checkpoint from {checkpoint_path}: {str(e)}")
        raise

class HParams():
    """Hyperparameters container class"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v
    
    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

def get_model_filenames(model_type):
    """
    Get the config and checkpoint filenames for a given model type
    
    Args:
        model_type (str): The model type (e.g., 'FreeVC', 'FreeVC-s', etc.)
        
    Returns:
        tuple: (config_filename, checkpoint_filename)
    """
    if model_type == "FreeVC (24kHz)":
        config_filename = "freevc-24.json"
        checkpoint_filename = "freevc-24.pth"
    else:
        base_name = model_type.lower().replace(' ', '-')
        config_filename = f"{base_name}.json"
        checkpoint_filename = f"{base_name}.pth"
            
    return config_filename, checkpoint_filename

def check_file_exists(filepath, description="File"):
    """Check if a file exists and raise a FileNotFoundError if it doesn't"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{description} not found: {filepath}")
    return True

def list_available_models(freevc_dir):
    """
    List available models by checking config and checkpoint files
    
    Args:
        freevc_dir (str): Path to the FreeVC directory
        
    Returns:
        list: List of available model types
    """
    available_models = []
    config_dir = os.path.join(freevc_dir, "configs")
    checkpoint_dir = os.path.join(freevc_dir, "checkpoints")
    
    if not os.path.exists(config_dir) or not os.path.exists(checkpoint_dir):
        print(f"Config or checkpoint directory not found: {config_dir}, {checkpoint_dir}")
        return available_models
    
    # Standard model types
    model_types = ["FreeVC", "FreeVC-s", "FreeVC (24kHz)"]
    
    for model_type in model_types:
        config_filename, checkpoint_filename = get_model_filenames(model_type)
        config_path = os.path.join(config_dir, config_filename)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        
        if os.path.exists(config_path) and os.path.exists(checkpoint_path):
            available_models.append(model_type)
            
    return available_models

def apply_post_processing(audio, sampling_rate, clarity_enhancement=0.3, 
                         voice_match_strength=0.5, presence_boost=0.4,
                         normalize_output=True, normalization_level=0.95):
    """
    Apply enhanced post-processing to improve the converted voice
    
    Args:
        audio (numpy.ndarray): Audio data to process
        sampling_rate (int): Sampling rate of the audio
        clarity_enhancement (float): Amount of clarity enhancement to apply (0.0-1.0)
        voice_match_strength (float): How strongly to enhance voice characteristics (0.0-1.0)
        presence_boost (float): Amount of vocal presence enhancement (0.0-1.0)
        normalize_output (bool): Whether to normalize the output audio
        normalization_level (float): Target level for normalization (0.0-1.0)
        
    Returns:
        numpy.ndarray: Processed audio data
    """
    try:
        # Gentle noise gate to remove background artifacts
        threshold = 0.005
        audio[np.abs(audio) < threshold] = 0
        
        # Apply voice matching EQ to enhance the voice characteristics
        if voice_match_strength > 0:
            # Enhance formants (vocal tract resonances) - key for voice mimicry
            # Different formant regions for better voice adaptation
            b1, a1 = signal.butter(2, [500/(sampling_rate/2), 1200/(sampling_rate/2)], 'bandpass', analog=False)
            formant1_enhanced = signal.filtfilt(b1, a1, audio)
            
            b2, a2 = signal.butter(2, [1200/(sampling_rate/2), 2500/(sampling_rate/2)], 'bandpass', analog=False)
            formant2_enhanced = signal.filtfilt(b2, a2, audio)
            
            # Higher formants for voice character
            b3, a3 = signal.butter(2, [2500/(sampling_rate/2), 3500/(sampling_rate/2)], 'bandpass', analog=False)
            formant3_enhanced = signal.filtfilt(b3, a3, audio)
            
            # Mix the enhancements with original with precise formant balancing
            audio = audio * (1-voice_match_strength) + \
                   formant1_enhanced * (voice_match_strength*0.5) + \
                   formant2_enhanced * (voice_match_strength*0.3) + \
                   formant3_enhanced * (voice_match_strength*0.2)
        
        # Enhance vocal presence - makes voice sound more "in the room"
        if presence_boost > 0:
            # Presence is typically in this region
            b_presence, a_presence = signal.butter(2, [2000/(sampling_rate/2), 5000/(sampling_rate/2)], 'bandpass', analog=False)
            presence = signal.filtfilt(b_presence, a_presence, audio)
            
            # Apply slight compression to presence band for more consistent character
            threshold = np.max(np.abs(presence)) * 0.3
            ratio = 0.7
            presence_compressed = np.where(
                np.abs(presence) > threshold,
                np.sign(presence) * (threshold + (np.abs(presence) - threshold) * ratio),
                presence
            )
            
            # Mix back with slight delay (1ms) for more depth
            delay_samples = int(sampling_rate * 0.001)
            if delay_samples > 0:
                delayed_presence = np.zeros_like(audio)
                delayed_presence[delay_samples:] = presence_compressed[:-delay_samples]
                audio = audio * (1-presence_boost) + delayed_presence * presence_boost
            else:
                audio = audio * (1-presence_boost) + presence_compressed * presence_boost
        
        # Apply clarity enhancement if requested - improves intelligibility
        if clarity_enhancement > 0:
            # Clarity lies in mid-high frequencies
            b, a = signal.butter(2, 3000/(sampling_rate/2), 'high', analog=False)
            clarity = signal.filtfilt(b, a, audio)
            
            # Add subtle harmonics enhancement (improves perceived clarity)
            harmonics = audio * audio * 0.1  # Simple harmonic generation
            harmonics = signal.filtfilt(b, a, harmonics)  # Filter to keep only high harmonics
            
            # Mix clarity enhancement with original
            audio = audio * (1-clarity_enhancement) + \
                   clarity * (clarity_enhancement * 0.8) + \
                   harmonics * (clarity_enhancement * 0.2)
                
        # Apply gentle multiband compression for professional sound
        try:
            # Low-end compression (below 300Hz)
            b_low, a_low = signal.butter(2, 300/(sampling_rate/2), 'low', analog=False)
            low_band = signal.filtfilt(b_low, a_low, audio)
            
            # Mid-band compression (300Hz-3kHz - where the voice mainly is)
            b_mid, a_mid = signal.butter(2, [300/(sampling_rate/2), 3000/(sampling_rate/2)], 'bandpass', analog=False)
            mid_band = signal.filtfilt(b_mid, a_mid, audio)
            
            # High-band compression (above 3kHz)
            b_high, a_high = signal.butter(2, 3000/(sampling_rate/2), 'high', analog=False)
            high_band = signal.filtfilt(b_high, a_high, audio)
            
            # Apply different compression settings to each band
            # Low band - stronger compression
            low_threshold = np.max(np.abs(low_band)) * 0.4
            low_ratio = 0.5
            low_compressed = np.where(
                np.abs(low_band) > low_threshold,
                np.sign(low_band) * (low_threshold + (np.abs(low_band) - low_threshold) * low_ratio),
                low_band
            )
            
            # Mid band - moderate compression
            mid_threshold = np.max(np.abs(mid_band)) * 0.5
            mid_ratio = 0.7
            mid_compressed = np.where(
                np.abs(mid_band) > mid_threshold,
                np.sign(mid_band) * (mid_threshold + (np.abs(mid_band) - mid_threshold) * mid_ratio),
                mid_band
            )
            
            # High band - gentle compression
            high_threshold = np.max(np.abs(high_band)) * 0.6
            high_ratio = 0.8
            high_compressed = np.where(
                np.abs(high_band) > high_threshold,
                np.sign(high_band) * (high_threshold + (np.abs(high_band) - high_threshold) * high_ratio),
                high_band
            )
            
            # Recombine the bands
            audio = low_compressed + mid_compressed + high_compressed
        except Exception as e:
            print(f"Multiband compression failed: {e}, skipping...")
        
        # Final gentle noise gate
        threshold_final = 0.01
        audio[np.abs(audio) < threshold_final] = 0
        
        # Apply output normalization if requested
        if normalize_output:
            print(f"Normalizing output audio to level: {normalization_level}")
            # First normalize to [-1, 1]
            if np.max(np.abs(audio)) > 0:
                audio = librosa.util.normalize(audio)
                # Then scale to the requested level
                audio = audio * normalization_level
                
        # Add very subtle warmth (low-mid emphasis)
        b_warm, a_warm = signal.butter(2, [100/(sampling_rate/2), 800/(sampling_rate/2)], 'bandpass', analog=False)
        warmth = signal.filtfilt(b_warm, a_warm, audio)
        audio = audio * 0.95 + warmth * 0.05
        
    except Exception as e:
        print(f"Post-processing failed: {e}, using unprocessed audio...")
            
    return audio

def process_audio(audio_data, target_sr, label=""):
    """
    Process audio data: convert to mono if stereo and resample if needed.
    
    Args:
        audio_data (dict): Audio data dictionary with 'waveform' and 'sample_rate'
        target_sr (int): Target sampling rate
        label (str): Label for logging purposes
    
    Returns:
        numpy.ndarray: Processed audio data
    """
    if not isinstance(audio_data, dict) or 'waveform' not in audio_data:
        raise ValueError(f"Invalid {label} audio format: {type(audio_data)}")
    
    # Extract waveform and handle stereo
    wav = audio_data['waveform'].squeeze().numpy()
    sr = audio_data['sample_rate']
    print(f"{label} audio loaded: {wav.shape}, sr={sr}")
    
    # Convert stereo to mono if needed
    if len(wav.shape) > 1 and wav.shape[0] == 2:
        print(f"Converting {label} audio from stereo to mono")
        wav = np.mean(wav, axis=0)
    
    # Resample if needed
    if sr != target_sr:
        print(f"Resampling {label} audio from {sr} to {target_sr}")
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    
    return wav

def preprocess_audio(wav, sr, target_sr, noise_reduction_strength=0.5, 
                    neutralize_source=0.0, enhance_clarity=0.0,
                    remove_sibilance=0.0, vad_sensitivity=20):
    """
    Enhanced preprocessing for audio with multiple options
    
    Args:
        wav (numpy.ndarray): Audio data
        sr (int): Sample rate of audio
        target_sr (int): Target sample rate
        noise_reduction_strength (float): Amount of noise reduction to apply
        neutralize_source (float): How much to neutralize voice characteristics
        enhance_clarity (float): Amount of clarity enhancement
        remove_sibilance (float): Amount of sibilance reduction
        vad_sensitivity (int): Voice activity detection sensitivity
        
    Returns:
        numpy.ndarray: Processed audio data
    """
    # Apply noise reduction
    if noise_reduction_strength > 0:
        try:
            wav = nr.reduce_noise(y=wav, sr=sr, prop_decrease=noise_reduction_strength)
        except Exception as e:
            print(f"Noise reduction failed: {e}, skipping...")
    
    # Voice activity detection to focus on speech segments
    try:
        intervals = librosa.effects.split(wav, top_db=vad_sensitivity)
        if len(intervals) > 0:  # Only apply if we found speech segments
            wav_output = np.zeros_like(wav)
            for interval in intervals:
                wav_output[interval[0]:interval[1]] = wav[interval[0]:interval[1]]
            wav = wav_output
    except Exception as e:
        print(f"VAD failed: {e}, using original audio...")
    
    # Neutralize source voice characteristics
    if neutralize_source > 0:
        try:
            # Flatten formants a bit to neutralize voice
            b1, a1 = signal.butter(2, [300/(sr/2), 3000/(sr/2)], 'bandpass', analog=False)
            neutral_voice = signal.filtfilt(b1, a1, wav)
            
            # Mix with original
            wav = wav * (1-neutralize_source) + neutral_voice * neutralize_source
        except Exception as e:
            print(f"Source neutralization failed: {e}, skipping...")
    
    # Enhance clarity
    if enhance_clarity > 0:
        try:
            b2, a2 = signal.butter(2, 2000/(sr/2), 'high', analog=False)
            clarity = signal.filtfilt(b2, a2, wav)
            wav = wav * (1-enhance_clarity) + clarity * enhance_clarity
        except Exception as e:
            print(f"Clarity enhancement failed: {e}, skipping...")
    
    # Reduce sibilance (de-essing)
    if remove_sibilance > 0:
        try:
            # Sibilance mainly in 5-8kHz range
            b3, a3 = signal.butter(2, [5000/(sr/2), 8000/(sr/2)], 'bandpass', analog=False)
            sibilance = signal.filtfilt(b3, a3, wav)
            
            # Compress sibilance
            thresh = np.max(np.abs(sibilance)) * 0.2
            ratio = 0.3
            sibilance_reduced = np.where(
                np.abs(sibilance) > thresh,
                np.sign(sibilance) * (thresh + (np.abs(sibilance) - thresh) * ratio),
                sibilance
            )
            
            # Recombine with reduced sibilance
            wav = wav - (sibilance * remove_sibilance) + (sibilance_reduced * remove_sibilance)
        except Exception as e:
            print(f"Sibilance reduction failed: {e}, skipping...")
    
    # Normalize audio
    wav = librosa.util.normalize(wav) * 0.95
    
    return wav

def process_multiple_references(reference_audios, target_sr, device, speaker_encoder, noise_reduction_strength=0.5):
    """
    Process multiple reference samples to get more robust speaker embedding
    
    Args:
        reference_audios (list): List of reference audio dictionaries
        target_sr (int): Target sampling rate
        device (torch.device): Device to use for processing
        speaker_encoder: Speaker encoder model
        noise_reduction_strength (float): Amount of noise reduction to apply
        
    Returns:
        torch.Tensor: Speaker embedding tensor
    """
    if not isinstance(reference_audios, list):
        reference_audios = [reference_audios]
            
    embeddings = []
    
    for ref_audio in reference_audios:
        wav_ref = process_audio(ref_audio, target_sr, "reference")
        wav_ref = preprocess_audio(wav_ref, target_sr, target_sr, noise_reduction_strength)
        
        # Get embedding for this reference
        try:
            print("Getting speaker embedding")
            ref_embed = speaker_encoder.embed_utterance(wav_ref)
            embeddings.append(ref_embed)
        except Exception as e:
            print(f"Error getting speaker embedding: {str(e)}")
                
    # Average the embeddings
    if embeddings:
        avg_embed = np.mean(embeddings, axis=0)
        # Re-normalize the averaged embedding
        avg_embed = avg_embed / np.linalg.norm(avg_embed, 2)
        return torch.from_numpy(avg_embed).unsqueeze(0).to(device)
    
    # If we get here with no embeddings, something went wrong
    print("WARNING: No embeddings generated from reference audio. This will cause errors.")
    return None

def save_audio(audio, file_path, sample_rate=16000):
    """
    Save audio data to a file
    
    Args:
        audio (numpy.ndarray): Audio data
        file_path (str): Path to save the audio file
        sample_rate (int): Sampling rate of the audio
    """
    try:
        # Normalize audio to avoid clipping
        audio = np.clip(audio, -1.0, 1.0)
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        # Save as WAV
        write(file_path, sample_rate, audio_int16)
        print(f"Audio saved to {file_path}")
    except Exception as e:
        print(f"Error saving audio: {e}")

# Simplified FreeVCNode class - moved to nodes.py

NODE_CLASS_MAPPINGS = {
    "FreeVC Voice Conversion": None  # Defined in nodes.py
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeVC Voice Conversion": "FreeVC Voice Converter 🎤"
}