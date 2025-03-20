# ComfyUI-FreeVC_wrapper

## Support My Work
If you find this project helpful, consider buying me a coffee:

[![Buy Me A Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=&slug=shmuelronen&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/shmuelronen)

A voice conversion extension node for ComfyUI based on [FreeVC](https://github.com/OlaWod/FreeVC), enabling high-quality voice conversion capabilities within the ComfyUI framework.

![image](https://github.com/user-attachments/assets/f4d2ef79-7910-4064-9813-3a628d589284)


## Features

- Support for multiple FreeVC models:
  - FreeVC: Standard 16kHz model
  - FreeVC-s: Source-filtering based model
  - FreeVC (24kHz): High-quality 24kHz model
- Stereo and mono audio support
- Automatic audio resampling
- Integrated with ComfyUI's audio processing pipeline
- GPU acceleration support (CUDA)

## Installation

1. Install the extension in your ComfyUI's custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ShmuelRonen/ComfyUI-FreeVC_wrapper.git
cd ComfyUI-FreeVC_wrapper
```

2. Install required Python packages:
```bash
pip install librosa transformers numpy torch
```

3. Download required checkpoints:

a. Voice Conversion Models:
Download the following checkpoint files from [HuggingFace](https://huggingface.co/spaces/OlaWod/FreeVC/tree/main/checkpoints) and place them in the `custom_nodes/ComfyUI-FreeVC_wrapper/freevc/checkpoints` directory:

| Model | Filename | Description |
|-------|----------|-------------|
| FreeVC | `freevc.pth` | Standard 16kHz model |
| FreeVC-s | `freevc-s.pth` | Source-filtering based model |
| FreeVC (24kHz) | `freevc-24.pth` | High-quality 24kHz model |

Direct download links:
- [freevc.pth](https://huggingface.co/spaces/OlaWod/FreeVC/resolve/main/checkpoints/freevc.pth)
- [freevc-s.pth](https://huggingface.co/spaces/OlaWod/FreeVC/resolve/main/checkpoints/freevc-s.pth)
- [freevc-24.pth](https://huggingface.co/spaces/OlaWod/FreeVC/resolve/main/checkpoints/freevc-24.pth)

b. Speaker Encoder:
Download the speaker encoder checkpoint from [HuggingFace](https://huggingface.co/spaces/OlaWod/FreeVC/tree/main/speaker_encoder/ckpt) and place it in the `custom_nodes/ComfyUI-FreeVC_wrapper/freevc/speaker_encoder/ckpt` directory:

| Component | Filename | Required For |
|-----------|----------|--------------|
| Speaker Encoder | `pretrained_bak_5805000.pt` | FreeVC and FreeVC (24kHz) models |

Direct download link:
- [pretrained_bak_5805000.pt](https://huggingface.co/spaces/OlaWod/FreeVC/resolve/main/speaker_encoder/ckpt/pretrained_bak_5805000.pt)


## Usage

1. In ComfyUI, locate the "FreeVC Voice Conversion" node under the "audio/voice conversion" category
2. Connect your inputs:
   - Source audio: The audio you want to convert
   - Reference audio: The target voice style
   - Select model type: Choose between FreeVC, FreeVC-s, or FreeVC (24kHz)
3. Connect the output to your desired audio output node

### Model Selection Guide

- **FreeVC**: Best for general purpose voice conversion at 16kHz
- **FreeVC-s**: Better preservation of source speech content, recommended for maintaining clarity
- **FreeVC (24kHz)**: Highest quality output with better audio fidelity

## Known Issues and Troubleshooting

1. **File Not Found Errors**:
   - Ensure all checkpoint files are in the correct directory
   - Verify file names match exactly: `freevc.pth`, `freevc-s.pth`, `freevc-24.pth`

2. **CUDA Out of Memory**:
   - Try processing shorter audio clips
   - Use CPU if GPU memory is insufficient

3. **Audio Format Issues**:
   - The node automatically handles stereo to mono conversion
   - Supports resampling from any sample rate
   - Trim silence from audio files for better results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original FreeVC implementation by [OlaWod](https://github.com/OlaWod/FreeVC)
- ComfyUI framework by [comfyanonymous](https://github.com/comfyanonymous/ComfyUI)

## Citation

If you use this in your research, please cite:
```bibtex
@article{wang2023freevc,
  title={FreeVC: Towards High-Quality Text-Free One-Shot Voice Conversion},
  author={Wang, Jiarui and Chen, Shilong and Wu, Yu and Zhang, Pan and Xie, Lei},
  journal={arXiv preprint arXiv:2210.15418},
  year={2023}
}
```
