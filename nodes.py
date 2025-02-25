import os
import sys
import json
import torch
import librosa
import numpy as np
from scipy.io.wavfile import write
from transformers import WavLMModel

# Add FreeVC directory to path
freevc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "freevc")
if freevc_dir not in sys.path:
    sys.path.append(freevc_dir)

# Import FreeVC modules
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from speaker_encoder.voice_encoder import SpeakerEncoder

# Local utils functions to replace the original
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
        """Initialize FreeVC Node with CUDA if available, otherwise CPU."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        print("FreeVC Node initialized on device:", self.device)

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
            # Load hparams
            # Handle filenames for different model types
            if model_type == "FreeVC (24kHz)":
                config_filename = "freevc-24.json"
                checkpoint_filename = "freevc-24.pth"
            else:
                config_filename = f"{model_type.lower().replace(' ', '-')}.json"
                checkpoint_filename = f"{model_type.lower().replace(' ', '-')}.pth"
            
            config_path = os.path.join(freevc_dir, "configs", config_filename)
            print(f"Loading config from: {config_path}")
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
            model, _, _ = load_checkpoint(checkpoint_path, model)
            
            # Load speaker encoder for specific models
            if model_type in ["FreeVC", "FreeVC (24kHz)"]:
                print("Loading speaker encoder...")
                speaker_encoder = SpeakerEncoder(os.path.join(freevc_dir, 'speaker_encoder/ckpt/pretrained_bak_5805000.pt'))
                self.models[model_type] = {
                    'model': model,
                    'speaker_encoder': speaker_encoder
                }
            else:
                self.models[model_type] = {
                    'model': model
                }
            
            # Load WavLM if not already loaded
            if 'wavlm' not in self.models:
                print("Loading WavLM...")
                print("Downloading WavLM model if needed...")
                self.models['wavlm'] = WavLMModel.from_pretrained("microsoft/wavlm-large").to(self.device)
                print("WavLM loaded successfully")
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise e

    @classmethod
    def INPUT_TYPES(s):
        """Define input types for ComfyUI."""
        return {
            "required": {
                "model_type": (["FreeVC", "FreeVC-s", "FreeVC (24kHz)"],),
                "source_audio": ("AUDIO",),
                "reference_audio": ("AUDIO",)
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "convert_voice"
    CATEGORY = "audio/voice conversion"

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

    def convert_voice(self, model_type, source_audio, reference_audio):
        """
        Convert voice using FreeVC models.
        
        Args:
            model_type (str): Type of FreeVC model to use
            source_audio (dict): Source audio data
            reference_audio (dict): Reference audio data
        
        Returns:
            tuple: Tuple containing dictionary with converted audio data
        """
        try:
            print("Starting voice conversion...")
            print(f"Processing with model: {model_type}")
            
            self.load_models(model_type)
            print("Models loaded successfully")
            
            with torch.no_grad():
                # Process reference audio
                print("Processing reference audio...")
                wav_tgt = self._process_audio(reference_audio, self.hps.data.sampling_rate, "reference")
                wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
                
                # Handle reference audio based on model type
                if model_type in ["FreeVC", "FreeVC (24kHz)"]:
                    print("Computing speaker embedding...")
                    g_tgt = self.models[model_type]['speaker_encoder'].embed_utterance(wav_tgt)
                    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(self.device)
                else:
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

                # Process source audio
                print("Processing source audio...")
                wav_src = self._process_audio(source_audio, self.hps.data.sampling_rate, "source")
                wav_src = torch.from_numpy(wav_src).unsqueeze(0).to(self.device)
                
                # Compute WavLM features
                print("Computing WavLM features...")
                c = self.models['wavlm'](wav_src).last_hidden_state.transpose(1, 2)
                print("WavLM features computed")

                # Convert voice based on model type
                print(f"Performing voice conversion with {model_type}...")
                if model_type == "FreeVC":
                    audio = self.models[model_type]['model'].infer(c, g=g_tgt)
                elif model_type == "FreeVC-s":
                    audio = self.models[model_type]['model'].infer(c, mel=mel_tgt)
                else:  # FreeVC (24kHz)
                    audio = self.models[model_type]['model'].infer(c, g=g_tgt)
                
                # Process output
                audio = audio[0][0].data.cpu().float().numpy()
                sampling_rate = 24000 if model_type == "FreeVC (24kHz)" else self.hps.data.sampling_rate
                print("Voice conversion completed")
                
                # Return in the same format as input
                print("Preparing output...")
                return ({"waveform": torch.tensor(audio).unsqueeze(0).unsqueeze(0), 
                        "sample_rate": sampling_rate},)
                
        except Exception as e:
            print(f"Error in voice conversion: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

NODE_CLASS_MAPPINGS = {
    "FreeVC Voice Conversion": FreeVCNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeVC Voice Converter": "FreeVC Voice Converter 🎤"
}