import os
import torch
import librosa
import gradio as gr
from scipy.io.wavfile import write
from transformers import WavLMModel

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from speaker_encoder.voice_encoder import SpeakerEncoder

'''
def get_wavlm():
    os.system('gdown https://drive.google.com/uc?id=12-cB34qCTvByWT-QtOcZaqwwO21FLSqU')
    shutil.move('WavLM-Large.pt', 'wavlm')
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading FreeVC...")
hps = utils.get_hparams_from_file("configs/freevc.json")
freevc = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).to(device)
_ = freevc.eval()
_ = utils.load_checkpoint("checkpoints/freevc.pth", freevc, None)
smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

print("Loading FreeVC(24k)...")
hps = utils.get_hparams_from_file("configs/freevc-24.json")
freevc_24 = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).to(device)
_ = freevc_24.eval()
_ = utils.load_checkpoint("checkpoints/freevc-24.pth", freevc_24, None)

print("Loading FreeVC-s...")
hps = utils.get_hparams_from_file("configs/freevc-s.json")
freevc_s = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).to(device)
_ = freevc_s.eval()
_ = utils.load_checkpoint("checkpoints/freevc-s.pth", freevc_s, None)

print("Loading WavLM for content...")
cmodel = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
 
def convert(model, src, tgt):
    with torch.no_grad():
        # tgt
        wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
        if model == "FreeVC" or model == "FreeVC (24kHz)":
            g_tgt = smodel.embed_utterance(wav_tgt)
            g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(device)
        else:
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).to(device)
            mel_tgt = mel_spectrogram_torch(
                wav_tgt, 
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
        # src
        wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
        wav_src = torch.from_numpy(wav_src).unsqueeze(0).to(device)
        c = cmodel(wav_src).last_hidden_state.transpose(1, 2).to(device)
        # infer
        if model == "FreeVC":
            audio = freevc.infer(c, g=g_tgt)
        elif model == "FreeVC-s":
            audio = freevc_s.infer(c, mel=mel_tgt)
        else:
            audio = freevc_24.infer(c, g=g_tgt)
        audio = audio[0][0].data.cpu().float().numpy()
        if model == "FreeVC" or model == "FreeVC-s":
            write("out.wav", hps.data.sampling_rate, audio)
        else:
            write("out.wav", 24000, audio)
    out = "out.wav"
    return out
    
model = gr.Dropdown(choices=["FreeVC", "FreeVC-s", "FreeVC (24kHz)"], value="FreeVC",type="value", label="Model") 
audio1 = gr.Audio(label="Source Audio", type='filepath')
audio2 = gr.Audio(label="Reference Audio", type='filepath')
inputs = [model, audio1, audio2]
outputs =  gr.Audio(label="Output Audio", type='filepath')

title = "FreeVC"
description = "Gradio Demo for FreeVC: Towards High-Quality Text-Free One-Shot Voice Conversion. To use it, simply upload your audio, or click the example to load. Read more at the links below. Note: It seems that the WavLM checkpoint in HuggingFace is a little different from the one used to train FreeVC, which may degrade the performance a bit. In addition, speaker similarity can be largely affected if there are too much silence in the reference audio, so please <strong>trim</strong> it before submitting."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2210.15418' target='_blank'>Paper</a> | <a href='https://github.com/OlaWod/FreeVC' target='_blank'>Github Repo</a></p>"

examples=[["FreeVC", 'p225_001.wav', 'p226_002.wav'], ["FreeVC-s", 'p226_002.wav', 'p225_001.wav'], ["FreeVC (24kHz)", 'p225_001.wav', 'p226_002.wav']]

iface = gr.Interface(fn=convert, inputs=inputs, outputs=outputs, title=title, description=description, article=article, examples=examples)
iface.queue().launch()