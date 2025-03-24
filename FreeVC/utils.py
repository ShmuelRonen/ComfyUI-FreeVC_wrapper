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

# Add FreeVC directory to path - MODIFY THIS PART
freevc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "freevc")
if freevc_dir not in sys.path:
    sys.path.insert(0, freevc_dir)  # Changed from append to insert(0)

# Import FreeVC modules - KEEP THESE AS THEY ARE
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
        model_type (str): The model type (e.g., 'FreeVC', 'D-FreeVC', etc.)
        
    Returns:
        tuple: (config_filename, checkpoint_filename)
    """
    is_diffusion = model_type.startswith("D-")
    base_model_type = model_type[2:] if is_diffusion else model_type
    
    if base_model_type == "FreeVC (24kHz)":
        if is_diffusion:
            config_filename = "d-freevc-24.json"
            checkpoint_filename = "d-freevc-24.pth"
        else:
            config_filename = "freevc-24.json"
            checkpoint_filename = "freevc-24.pth"
    else:
        base_name = base_model_type.lower().replace(' ', '-')
        if is_diffusion:
            config_filename = f"d-{base_name}.json"
            checkpoint_filename = f"d-{base_name}.pth"
        else:
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
    model_types = ["FreeVC", "FreeVC-s", "FreeVC (24kHz)", "D-FreeVC", "D-FreeVC-s", "D-FreeVC (24kHz)"]
    
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
            # The ratios here are based on typical importance of different formant regions
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

def process_multiple_references(reference_audios, model_type, target_sr, device, speaker_encoder, noise_reduction_strength=0.5):
    """
    Process multiple reference samples to get more robust speaker embedding
    
    Args:
        reference_audios (list): List of reference audio dictionaries
        model_type (str): The model type
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
    base_model_type = model_type[2:] if model_type.startswith("D-") else model_type
    
    for ref_audio in reference_audios:
        wav_ref = process_audio(ref_audio, target_sr, "reference")
        wav_ref = preprocess_audio(wav_ref, target_sr, target_sr, noise_reduction_strength)
        
        # Get embedding for this reference
        if base_model_type in ["FreeVC", "FreeVC (24kHz)"]:
            try:
                print(f"Getting speaker embedding for {model_type}")
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

def apply_post_processing(audio, sampling_rate, clarity_enhancement=0.3, normalize_output=True, normalization_level=0.95):
    """
    Apply post-processing to enhance the converted voice
    
    Args:
        audio (numpy.ndarray): Audio data to process
        sampling_rate (int): Sampling rate of the audio
        clarity_enhancement (float): Amount of clarity enhancement to apply (0.0-1.0)
        normalize_output (bool): Whether to normalize the output audio
        normalization_level (float): Target level for normalization (0.0-1.0)
        
    Returns:
        numpy.ndarray: Processed audio data
    """
    try:
        # Gentle noise gate to remove background artifacts
        threshold = 0.005
        audio[np.abs(audio) < threshold] = 0
        
        # Apply voice matching EQ if requested
        if voice_match_strength > 0:
            # Enhance formants (vocal tract resonances) - key for voice mimicry
            b1, a1 = signal.butter(2, [500/(sampling_rate/2), 3500/(sampling_rate/2)], 'bandpass', analog=False)
            formant_enhanced = signal.filtfilt(b1, a1, audio)
            
            # Boost around 1-2kHz range where voice character often resides
            b2, a2 = signal.butter(2, [1000/(sampling_rate/2), 2000/(sampling_rate/2)], 'bandpass', analog=False)
            character_enhanced = signal.filtfilt(b2, a2, audio)
            
            # Mix the enhancements with original
            audio = audio * (1-voice_match_strength) + formant_enhanced * (voice_match_strength*0.6) + character_enhanced * (voice_match_strength*0.4)
        
        # Apply clarity enhancement if requested
        if clarity_enhancement > 0:
            b, a = signal.butter(2, 3000/(sampling_rate/2), 'high', analog=False)
            audio = signal.filtfilt(b, a, audio) * clarity_enhancement + audio * (1-clarity_enhancement)
            
        # Apply output normalization if requested
        if normalize_output:
            if np.max(np.abs(audio)) > 0:
                audio = librosa.util.normalize(audio)
                audio = audio * normalization_level
    except Exception as e:
        print(f"Post-processing failed: {e}, using unprocessed audio...")
            
    return audio

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

def modify_diffusion_params(hps, noise_coef=None, steps=None):
    """
    Modify diffusion parameters in the hyperparameters
    
    Args:
        hps (HParams): Hyperparameters object
        noise_coef (float, optional): New noise coefficient value
        steps (int, optional): New diffusion steps value
        
    Returns:
        tuple: (original_noise_coef, original_steps) - Original values for restoration
    """
    if not hasattr(hps, 'train') or not hasattr(hps.train, 'diffusion'):
        return None, None
    
    # Save original values
    original_noise_coef = hps.train.diffusion.get('inference_noise_coef', 0.3)
    original_steps = hps.train.diffusion.get('diffusion_steps', 50)
    
    # Update values if provided
    if noise_coef is not None:
        hps.train.diffusion['inference_noise_coef'] = noise_coef
    
    if steps is not None:
        hps.train.diffusion['diffusion_steps'] = steps
    
    return original_noise_coef, original_steps

class FreeVCNode:
    def __init__(self):
        """Initialize Enhanced FreeVC Node with improved processing."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.current_model_type = None
        print("Enhanced FreeVC Node initialized on device:", self.device)
    
    def load_models(self, model_type):
        """
        Load required models for voice conversion if not already loaded.
        
        Args:
            model_type (str): Type of FreeVC model to load
        """
        if model_type in self.models:
            return
            
        print(f"Loading {model_type}...")
        try:
            # Check if this is a diffusion model variant
            is_diffusion = model_type.startswith("D-")
            base_model_type = model_type[2:] if is_diffusion else model_type
            
            # Get filenames for this model type
            config_filename, checkpoint_filename = get_model_filenames(model_type)
            
            # Print exact filenames for debugging
            print(f"Using config file: {config_filename}")
            print(f"Using checkpoint file: {checkpoint_filename}")
            
            config_path = os.path.join(freevc_dir, "configs", config_filename)
            print(f"Loading config from: {config_path}")
            
            # Check if config file exists
            check_file_exists(config_path, "Config file")
                
            self.hps = load_config(config_path)
            
            # Initialize synthesis model
            print("Initializing synthesis model...")
            model = SynthesizerTrn(
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                **self.hps.model).to(self.device)
            model.eval()
            
            # Load checkpoint
            checkpoint_path = os.path.join(freevc_dir, "checkpoints", checkpoint_filename)
            print(f"Loading checkpoint from: {checkpoint_path}")
            
            # Check if checkpoint file exists
            check_file_exists(checkpoint_path, "Checkpoint file")
            
            # Try to load the checkpoint
            try:
                model, _, _ = load_checkpoint(checkpoint_path, model)
                print(f"Successfully loaded checkpoint: {checkpoint_filename}")
                
                # Load speaker encoder for specific models
                if base_model_type in ["FreeVC", "FreeVC (24kHz)"]:
                    print("Loading speaker encoder...")
                    encoder_path = os.path.join(freevc_dir, 'speaker_encoder/ckpt/pretrained_bak_5805000.pt')
                    
                    # Check if speaker encoder exists
                    check_file_exists(encoder_path, "Speaker encoder checkpoint")
                        
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()
                    
                    speaker_encoder = SpeakerEncoder(encoder_path)
                    
                    end_time.record()
                    torch.cuda.synchronize()
                    print(f"Loaded the voice encoder model on {self.device} in {start_time.elapsed_time(end_time)/1000:.2f} seconds.")
                    
                    self.models[model_type] = {
                        'model': model,
                        'speaker_encoder': speaker_encoder,
                        'is_diffusion': is_diffusion
                    }
                else:
                    self.models[model_type] = {
                        'model': model,
                        'is_diffusion': is_diffusion
                    }
            except Exception as e:
                print(f"Error loading {checkpoint_filename}: {str(e)}")
                if is_diffusion:
                    print(f"Falling back to standard model: {base_model_type}")
                    return self.load_models(base_model_type)
                else:
                    raise e
                
            # Load WavLM if not already loaded
            if 'wavlm' not in self.models:
                print("Loading WavLM...")
                print("Downloading WavLM model if needed...")
                self.models['wavlm'] = WavLMModel.from_pretrained("microsoft/wavlm-large").to(self.device)
                print("WavLM loaded successfully")
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fall back to standard model if diffusion model fails
            if is_diffusion:
                print(f"Falling back to standard model: {base_model_type}")
                return self.load_models(base_model_type)
            else:
                raise e
    
    @classmethod
    def INPUT_TYPES(s):
        """Define enhanced input types for ComfyUI."""
        return {
            "required": {
                "model_type": (["FreeVC", "FreeVC-s", "FreeVC (24kHz)", 
                               "D-FreeVC", "D-FreeVC-s", "D-FreeVC (24kHz)"],),
                "source_audio": ("AUDIO",),
                "reference_audio": ("AUDIO",)
            },
            "optional": {
                "secondary_reference": ("AUDIO", {"default": None}),
                
                # Source audio processing
                "noise_reduction_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "neutralize_source": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.7, "step": 0.1}),
                "enhance_clarity": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.7, "step": 0.1}),
                "remove_sibilance": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.7, "step": 0.1}),
                "vad_sensitivity": ("INT", {"default": 20, "min": 10, "max": 40, "step": 5}),
                
                # Conversion parameters
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
                "diffusion_noise_coef": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.05}),
                "diffusion_steps": ("INT", {"default": 30, "min": 10, "max": 50, "step": 5}),
                
                # Post-processing
                "clarity_enhancement": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "voice_match_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "presence_boost": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 0.8, "step": 0.1}),
                "normalize_output": ("BOOLEAN", {"default": True}),
                "normalization_level": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05})
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "convert_voice_enhanced"
    CATEGORY = "audio/voice conversion"

    def convert_voice_enhanced(self, model_type, source_audio, reference_audio, 
                             secondary_reference=None, 
                             # Source processing parameters
                             noise_reduction_strength=0.5, 
                             neutralize_source=0.3,
                             enhance_clarity=0.2,
                             remove_sibilance=0.3,
                             vad_sensitivity=20,
                             # Conversion parameters
                             temperature=0.6,
                             diffusion_noise_coef=0.05, 
                             diffusion_steps=30,
                             # Post-processing parameters
                             clarity_enhancement=0.3, 
                             voice_match_strength=0.5,
                             presence_boost=0.4,
                             normalize_output=True, 
                             normalization_level=0.95):
        """
        Enhanced voice conversion with multiple references and improved processing
        
        Args:
            model_type (str): FreeVC model type to use
            source_audio (dict): Source audio to convert
            reference_audio (dict): Reference audio for voice characteristics
            secondary_reference (dict, optional): Secondary reference for more robust conversion
            
            # Source processing parameters
            noise_reduction_strength (float): Amount of noise reduction to apply (0.0-1.0)
            neutralize_source (float): How much to neutralize source voice characteristics (0.0-0.7)
            enhance_clarity (float): Amount of speech clarity enhancement in preprocessing (0.0-0.7)
            remove_sibilance (float): Amount of sibilance (ssss sounds) reduction (0.0-0.7)
            vad_sensitivity (int): Voice activity detection sensitivity (lower = more sensitive)
            
            # Conversion parameters
            temperature (float): Temperature for WavLM feature extraction (0.1-1.0)
            diffusion_noise_coef (float): Noise coefficient for diffusion models (0.0-0.5)
            diffusion_steps (int): Number of diffusion steps (10-50)
            
            # Post-processing parameters
            clarity_enhancement (float): Amount of clarity enhancement to apply (0.0-1.0)
            voice_match_strength (float): How strongly to enhance voice characteristics (0.0-1.0)
            presence_boost (float): Amount of vocal presence enhancement (0.0-0.8)
            normalize_output (bool): Whether to normalize the output audio
            normalization_level (float): Target level for normalization (0.0-1.0)
            
        Returns:
            tuple: Audio data in ComfyUI format
        """
        try:
            print(f"Starting enhanced voice conversion with {model_type}...")
            
            self.current_model_type = model_type
            self.load_models(model_type)
            print("Models loaded successfully")
            
            # Check if using diffusion model
            is_diffusion = self.models[model_type].get('is_diffusion', False)
            base_model_type = model_type[2:] if is_diffusion else model_type
            
            print(f"Using {'diffusion-based' if is_diffusion else 'standard'} model: {model_type}")
            
            # Apply custom diffusion parameters if needed
            original_noise_coef, original_steps = None, None
            if is_diffusion and hasattr(self.hps, 'train') and hasattr(self.hps.train, 'diffusion'):
                print(f"Setting diffusion parameters - noise_coef: {diffusion_noise_coef}, steps: {diffusion_steps}")
                original_noise_coef = self.hps.train.diffusion.get('inference_noise_coef', 0.3)
                original_steps = self.hps.train.diffusion.get('diffusion_steps', 50)
                
                # Apply user settings
                self.hps.train.diffusion['inference_noise_coef'] = diffusion_noise_coef
                self.hps.train.diffusion['diffusion_steps'] = diffusion_steps
            
            with torch.no_grad():
                # Process source audio with enhanced preprocessing
                print("Processing source audio...")
                wav_src = self._process_audio(source_audio, self.hps.data.sampling_rate, "source")
                
                # Apply enhanced preprocessing with new parameters
                wav_src = self._preprocess_audio(
                    wav_src, 
                    self.hps.data.sampling_rate, 
                    self.hps.data.sampling_rate, 
                    noise_reduction_strength=noise_reduction_strength,
                    neutralize_source=neutralize_source,
                    enhance_clarity=enhance_clarity,
                    remove_sibilance=remove_sibilance,
                    vad_sensitivity=vad_sensitivity
                )
                
                wav_src = torch.from_numpy(wav_src).unsqueeze(0).to(self.device)
                
                # Create reference list and process reference audio(s)
                if base_model_type in ["FreeVC", "FreeVC (24kHz)"]:
                    # Process reference embeddings
                    ref_list = [reference_audio]
                    if secondary_reference is not None:
                        ref_list.append(secondary_reference)
                        
                    print("Processing reference audio for speaker embedding...")
                    g_tgt = None
                    for ref_audio in ref_list:
                        wav_ref = self._process_audio(ref_audio, self.hps.data.sampling_rate, "reference")
                        # For reference audio, we don't neutralize source but apply other enhancements
                        wav_ref = self._preprocess_audio(
                            wav_ref, 
                            self.hps.data.sampling_rate, 
                            self.hps.data.sampling_rate, 
                            noise_reduction_strength=noise_reduction_strength,
                            neutralize_source=0.0,  # Don't neutralize reference
                            enhance_clarity=enhance_clarity,
                            remove_sibilance=remove_sibilance,
                            vad_sensitivity=vad_sensitivity
                        )
                        
                        # Get embedding for this reference
                        try:
                            # Make sure speaker encoder exists for current model
                            if 'speaker_encoder' not in self.models[self.current_model_type]:
                                print(f"Warning: Speaker encoder not found for {self.current_model_type}")
                                continue
                                
                            print(f"Getting speaker embedding for {self.current_model_type}")
                            ref_embed = self.models[self.current_model_type]['speaker_encoder'].embed_utterance(wav_ref)
                            
                            if g_tgt is None:
                                g_tgt = torch.from_numpy(ref_embed).unsqueeze(0).to(self.device)
                            else:
                                # Average with previous embeddings
                                new_embed = torch.from_numpy(ref_embed).unsqueeze(0).to(self.device)
                                g_tgt = (g_tgt + new_embed) / 2.0
                        except Exception as e:
                            print(f"Error getting speaker embedding: {str(e)}")
                    
                    # Safety check - fall back to standard model if we don't get a valid g_tgt
                    if g_tgt is None and is_diffusion:
                        print(f"Failed to get speaker embedding. Falling back to standard model: {base_model_type}")
                        
                        # Restore original diffusion parameters if modified
                        if original_noise_coef is not None and original_steps is not None:
                            self.hps.train.diffusion['inference_noise_coef'] = original_noise_coef
                            self.hps.train.diffusion['diffusion_steps'] = original_steps
                            
                        return self.convert_voice_enhanced(
                            base_model_type, source_audio, reference_audio,
                            secondary_reference, noise_reduction_strength,
                            neutralize_source, enhance_clarity, remove_sibilance,
                            vad_sensitivity, temperature, diffusion_noise_coef,
                            diffusion_steps, clarity_enhancement, voice_match_strength,
                            presence_boost, normalize_output, normalization_level
                        )
                else:
                    # For FreeVC-s, process the primary reference for mel spectrogram
                    print("Processing reference audio for mel...")
                    wav_tgt = self._process_audio(reference_audio, self.hps.data.sampling_rate, "reference")
                    wav_tgt = self._preprocess_audio(
                        wav_tgt, 
                        self.hps.data.sampling_rate, 
                        self.hps.data.sampling_rate, 
                        noise_reduction_strength=noise_reduction_strength,
                        neutralize_source=0.0,  # Don't neutralize reference
                        enhance_clarity=enhance_clarity,
                        remove_sibilance=remove_sibilance,
                        vad_sensitivity=vad_sensitivity
                    )
                    
                    # Add a quality check for reference audio
                    if np.max(np.abs(wav_tgt)) < 0.1:
                        print("Warning: Reference audio is very quiet, this may affect conversion quality")
                    
                    print("Computing mel spectrogram...")
                    wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).to(self.device)
                    mel_tgt = mel_spectrogram_torch(
                        wav_tgt, 
                        self.hps.data.filter_length,
                        self.hps.data.n_mel_channels,
                        self.hps.data.sampling_rate,
                        self.hps.data.hop_length,
                        self.hps.data.win_length,
                        self.hps.data.mel_fmin,
                        self.hps.data.mel_fmax
                    )
                
                # Compute WavLM features with temperature control
                print("Computing WavLM features...")
                c = self.models['wavlm'](wav_src).last_hidden_state
                
                # Apply temperature to control the feature extraction process
                if temperature < 1.0:
                    print(f"Applying temperature {temperature} to WavLM features")
                    c = c / temperature
                c = c.transpose(1, 2)
                
                # Convert voice based on model type
                print(f"Performing voice conversion with {model_type}...")
                    
                # Perform the actual voice conversion
                if base_model_type in ["FreeVC", "FreeVC (24kHz)"]:
                    audio = self.models[model_type]['model'].infer(c, g=g_tgt)
                else:  # FreeVC-s
                    audio = self.models[model_type]['model'].infer(c, mel=mel_tgt)
                
                # Process output
                audio = audio[0][0].data.cpu().float().numpy()
                
                # Determine sampling rate
                sampling_rate = 24000 if base_model_type == "FreeVC (24kHz)" else self.hps.data.sampling_rate
                
                # Apply enhanced post-processing with voice matching
                audio = self._apply_post_processing(
                    audio, 
                    sampling_rate, 
                    clarity_enhancement=clarity_enhancement,
                    voice_match_strength=voice_match_strength,
                    presence_boost=presence_boost,
                    normalize_output=normalize_output,
                    normalization_level=normalization_level
                )
                
                # Restore original diffusion parameters if modified
                if original_noise_coef is not None and original_steps is not None:
                    self.hps.train.diffusion['inference_noise_coef'] = original_noise_coef
                    self.hps.train.diffusion['diffusion_steps'] = original_steps
                
                print("Enhanced voice conversion completed")
                
                # Return in the same format as input
                print("Preparing output...")
                return ({"waveform": torch.tensor(audio).unsqueeze(0).unsqueeze(0), 
                        "sample_rate": sampling_rate},)
                    
        except Exception as e:
            print(f"Error in voice conversion: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fall back to standard model if diffusion model fails
            if is_diffusion:
                print(f"Falling back to standard model: {base_model_type}")
                return self.convert_voice_enhanced(
                    base_model_type, source_audio, reference_audio,
                    secondary_reference, noise_reduction_strength,
                    neutralize_source, enhance_clarity, remove_sibilance,
                    vad_sensitivity, temperature, diffusion_noise_coef,
                    diffusion_steps, clarity_enhancement, voice_match_strength,
                    presence_boost, normalize_output, normalization_level
                )
            else:
                raise e

NODE_CLASS_MAPPINGS = {
    "FreeVC Voice Conversion": FreeVCNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeVC Voice Conversion": "FreeVC Voice Converter v2 🎤"
}