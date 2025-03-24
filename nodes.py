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

# Reuse the existing utility functions
def load_config(config_path):
    with open(config_path, 'r') as f:
        data = f.read()
    config = json.loads(data)
    return HParams(**config)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
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

class HParams():
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
            
            # Handle filenames for different model types
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
            
            # Print exact filenames for debugging
            print(f"Using config file: {config_filename}")
            print(f"Using checkpoint file: {checkpoint_filename}")
            
            config_path = os.path.join(freevc_dir, "configs", config_filename)
            print(f"Loading config from: {config_path}")
            
            # Check if config file exists
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
                
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
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            # Try to load the checkpoint
            try:
                model, _, _ = load_checkpoint(checkpoint_path, model)
                print(f"Successfully loaded checkpoint: {checkpoint_filename}")
                
                # Load speaker encoder for specific models
                if base_model_type in ["FreeVC", "FreeVC (24kHz)"]:
                    print("Loading speaker encoder...")
                    encoder_path = os.path.join(freevc_dir, 'speaker_encoder/ckpt/pretrained_bak_5805000.pt')
                    
                    # Check if speaker encoder exists
                    if not os.path.exists(encoder_path):
                        raise FileNotFoundError(f"Speaker encoder checkpoint not found: {encoder_path}")
                        
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
    
    def _preprocess_audio(self, wav, sr, target_sr, noise_reduction_strength=0.5):
        """
        Enhanced preprocessing with noise reduction and VAD
        """
        # Apply noise reduction
        if noise_reduction_strength > 0:
            try:
                wav = nr.reduce_noise(y=wav, sr=sr, prop_decrease=noise_reduction_strength)
            except Exception as e:
                print(f"Noise reduction failed: {e}, skipping...")
            
        # Voice activity detection to focus on speech segments
        try:
            intervals = librosa.effects.split(wav, top_db=20)
            if len(intervals) > 0:  # Only apply if we found speech segments
                wav_output = np.zeros_like(wav)
                for interval in intervals:
                    wav_output[interval[0]:interval[1]] = wav[interval[0]:interval[1]]
                wav = wav_output
        except Exception as e:
            print(f"VAD failed: {e}, using original audio...")
        
        # Normalize audio
        wav = librosa.util.normalize(wav) * 0.95
        
        return wav
        
    def _process_audio(self, audio_data, target_sr, label=""):
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
    
    def _process_multiple_references(self, reference_audios, target_sr, noise_reduction_strength):
        """
        Process multiple reference samples to get more robust speaker embedding
        """
        if not isinstance(reference_audios, list):
            reference_audios = [reference_audios]
            
        embeddings = []
        base_model_type = self.current_model_type[2:] if self.current_model_type.startswith("D-") else self.current_model_type
        
        for ref_audio in reference_audios:
            wav_ref = self._process_audio(ref_audio, target_sr, "reference")
            wav_ref = self._preprocess_audio(wav_ref, target_sr, target_sr, noise_reduction_strength)
            
            # Get embedding for this reference - Note: use base_model_type for check
            if base_model_type in ["FreeVC", "FreeVC (24kHz)"]:
                try:
                    # Make sure speaker encoder exists for current model
                    if 'speaker_encoder' not in self.models[self.current_model_type]:
                        print(f"Warning: Speaker encoder not found for {self.current_model_type}")
                        continue
                        
                    print(f"Getting speaker embedding for {self.current_model_type}")
                    ref_embed = self.models[self.current_model_type]['speaker_encoder'].embed_utterance(wav_ref)
                    embeddings.append(ref_embed)
                except Exception as e:
                    print(f"Error getting speaker embedding: {str(e)}")
                
        # Average the embeddings
        if embeddings:
            avg_embed = np.mean(embeddings, axis=0)
            # Re-normalize the averaged embedding
            avg_embed = avg_embed / np.linalg.norm(avg_embed, 2)
            return torch.from_numpy(avg_embed).unsqueeze(0).to(self.device)
            
        # If we get here with no embeddings, something went wrong
        print("WARNING: No embeddings generated from reference audio. This will cause errors.")
        return None
    
    def _apply_post_processing(self, audio, sampling_rate, clarity_enhancement=0.3, normalize_output=True, normalization_level=0.95):
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
            
            # Apply slight EQ to enhance clarity if requested
            # Simple high-shelf filter to improve intelligibility
            if clarity_enhancement > 0:
                b, a = signal.butter(2, 3000/(sampling_rate/2), 'high', analog=False)
                audio = signal.filtfilt(b, a, audio) * clarity_enhancement + audio * (1-clarity_enhancement)
                
            # Apply output normalization if requested
            if normalize_output:
                print(f"Normalizing output audio to level: {normalization_level}")
                # First normalize to [-1, 1]
                if np.max(np.abs(audio)) > 0:
                    audio = librosa.util.normalize(audio)
                    # Then scale to the requested level
                    audio = audio * normalization_level
        except Exception as e:
            print(f"Post-processing failed: {e}, using unprocessed audio...")
            
        return audio
        
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
                "noise_reduction_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "clarity_enhancement": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "diffusion_noise_coef": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05}),
                "diffusion_steps": ("INT", {"default": 20, "min": 10, "max": 50, "step": 5}),
                "normalize_output": ("BOOLEAN", {"default": True}),
                "normalization_level": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05})
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "convert_voice_enhanced"
    CATEGORY = "audio/voice conversion"

    def convert_voice_enhanced(self, model_type, source_audio, reference_audio, 
                              secondary_reference=None, noise_reduction_strength=0.5, 
                              clarity_enhancement=0.3, temperature=0.7,
                              diffusion_noise_coef=0.1, diffusion_steps=20,
                              normalize_output=True, normalization_level=0.95):
        """
        Enhanced voice conversion with multiple references and improved processing
        
        Args:
            model_type (str): FreeVC model type to use
            source_audio (dict): Source audio to convert
            reference_audio (dict): Reference audio for voice characteristics
            secondary_reference (dict, optional): Secondary reference for more robust conversion
            noise_reduction_strength (float): Amount of noise reduction to apply (0.0-1.0)
            clarity_enhancement (float): Amount of clarity enhancement to apply (0.0-1.0)
            temperature (float): Temperature for WavLM feature extraction (0.1-1.0)
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
            
            with torch.no_grad():
                # Process source audio with enhanced preprocessing
                print("Processing source audio...")
                wav_src = self._process_audio(source_audio, self.hps.data.sampling_rate, "source")
                wav_src = self._preprocess_audio(wav_src, self.hps.data.sampling_rate, 
                                                self.hps.data.sampling_rate, noise_reduction_strength)
                wav_src = torch.from_numpy(wav_src).unsqueeze(0).to(self.device)
                
                # Create reference list and process reference audio(s)
                if base_model_type in ["FreeVC", "FreeVC (24kHz)"]:
                    # Process reference embeddings
                    ref_list = [reference_audio]
                    if secondary_reference is not None:
                        ref_list.append(secondary_reference)
                    g_tgt = self._process_multiple_references(ref_list, self.hps.data.sampling_rate, 
                                                             noise_reduction_strength)
                    
                    # Safety check - fall back to standard model if we don't get a valid g_tgt
                    if g_tgt is None and is_diffusion:
                        print(f"Failed to get speaker embedding. Falling back to standard model: {base_model_type}")
                        return self.convert_voice_enhanced(base_model_type, source_audio, reference_audio,
                                                        secondary_reference, noise_reduction_strength,
                                                        clarity_enhancement, temperature,
                                                        normalize_output, normalization_level)
                else:
                    # For FreeVC-s, process the primary reference for mel spectrogram
                    print("Processing reference audio for mel...")
                    wav_tgt = self._process_audio(reference_audio, self.hps.data.sampling_rate, "reference")
                    wav_tgt = self._preprocess_audio(wav_tgt, self.hps.data.sampling_rate, 
                                                   self.hps.data.sampling_rate, noise_reduction_strength)
                    wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
                    
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
                    c = c / temperature
                c = c.transpose(1, 2)
                
                # Convert voice based on model type
                print(f"Performing voice conversion with {model_type}...")
                # Apply custom diffusion parameters for diffusion-based models
                if is_diffusion:
                    print(f"Using diffusion parameters - noise_coef: {diffusion_noise_coef}, steps: {diffusion_steps}")
                    if self.hps.train.get('diffusion'):
                        # Temporarily save original settings to restore later
                        original_noise_coef = self.hps.train.diffusion.get('inference_noise_coef', 0.3)
                        original_steps = self.hps.train.diffusion.get('diffusion_steps', 50)
                        
                        # Apply user settings
                        self.hps.train.diffusion['inference_noise_coef'] = diffusion_noise_coef
                        self.hps.train.diffusion['diffusion_steps'] = diffusion_steps
                
                # Perform voice conversion based on model type
                if base_model_type in ["FreeVC", "FreeVC (24kHz)"]:
                    audio = self.models[model_type]['model'].infer(c, g=g_tgt)
                else:  # FreeVC-s
                    audio = self.models[model_type]['model'].infer(c, mel=mel_tgt)
                    
                # Restore original diffusion settings if modified
                if is_diffusion and self.hps.train.get('diffusion'):
                    self.hps.train.diffusion['inference_noise_coef'] = original_noise_coef
                    self.hps.train.diffusion['diffusion_steps'] = original_steps
                
                # Process output
                audio = audio[0][0].data.cpu().float().numpy()
                
                # Determine sampling rate
                sampling_rate = 24000 if base_model_type == "FreeVC (24kHz)" else self.hps.data.sampling_rate
                
                # Apply enhanced post-processing with normalization
                audio = self._apply_post_processing(
                    audio, 
                    sampling_rate, 
                    clarity_enhancement,
                    normalize_output,
                    normalization_level
                )
                
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
                return self.convert_voice_enhanced(base_model_type, source_audio, reference_audio,
                                                 secondary_reference, noise_reduction_strength,
                                                 clarity_enhancement, temperature,
                                                 normalize_output, normalization_level)
            else:
                raise e

NODE_CLASS_MAPPINGS = {
    "FreeVC Voice Conversion": FreeVCNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeVC Voice Conversion": "FreeVC Voice Converter v2 🎤"
}