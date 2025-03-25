# ComfyUI-FreeVC_wrapper

## Support My Work
If you find this project helpful, consider buying me a coffee:

[![Buy Me A Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=&slug=shmuelronen&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/shmuelronen)

A voice conversion extension node for ComfyUI based on [FreeVC](https://github.com/OlaWod/FreeVC), enabling high-quality voice conversion capabilities within the ComfyUI framework.

![image](https://github.com/user-attachments/assets/296863cc-7c71-458a-9794-f7cf0f0a4290)

## Features

- Support for multiple FreeVC models:
  - Standard models (16kHz): FreeVC, FreeVC-s
  - High-quality model (24kHz): FreeVC (24kHz) 
 - Enhanced voice mimicry capabilities
- Advanced audio pre and post-processing options
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
pip install librosa transformers numpy torch noisereduce
```

3. Download required checkpoints:

a. **Voice Conversion Models**:
All model checkpoint files (3 models) are available in a single Google Drive folder:
[Download All Model Checkpoints (Google Drive)](https://drive.google.com/file/d/1uGdFgjGV_rKlF0CyTToTRxhDsR0Y-dl_/view?usp=sharing)

After downloading, extract the file and place the checkpoints folder in the freevc directory:
```
ComfyUI-FreeVC_wrapper/freevc/
```

b. **Speaker Encoder**: 
Download the speaker encoder checkpoint from [HuggingFace](https://huggingface.co/spaces/OlaWod/FreeVC/tree/main/speaker_encoder/ckpt) and place it in the `custom_nodes/ComfyUI-FreeVC_wrapper/freevc/speaker_encoder/ckpt` directory:

| Component | Filename | Required For |
|-----------|----------|--------------|
| Speaker Encoder | `pretrained_bak_5805000.pt` | FreeVC, FreeVC (24kHz), D-FreeVC, and D-FreeVC (24kHz) models |

Direct download link:
- [pretrained_bak_5805000.pt](https://huggingface.co/spaces/OlaWod/FreeVC/resolve/main/speaker_encoder/ckpt/pretrained_bak_5805000.pt)

Your final directory structure should look like this:
```
ComfyUI-FreeVC_wrapper/
├── freevc/
    ├── checkpoints/
    │   ├── freevc.pth         # Standard 16kHz model
    │   ├── freevc-s.pth       # Source-filtering based model
    │   ├── freevc-24.pth      # High-quality 24kHz model
    │  
    └── speaker_encoder/
        └── ckpt/
            └── pretrained_bak_5805000.pt  # Speaker encoder checkpoint
```

## Usage

1. In ComfyUI, locate the "FreeVC Voice Converter v2 🎤" node under the "audio/voice conversion" category
2. Connect your inputs:
   - Source audio: The audio you want to convert
   - Reference audio: The target voice style
   - (Optional) Secondary reference: Additional reference for more robust voice matching
   - Select model type: Choose between standard and diffusion-enhanced models

3. Configure the conversion parameters:
   - Source processing: Noise reduction, source neutralization, clarity enhancement
   - Conversion settings: Temperature, diffusion parameters (for diffusion models)
   - Post-processing: Voice matching strength, presence boost, normalization

4. Connect the output to your desired audio output node

### Model Selection Guide

- **FreeVC**: Good for general purpose voice conversion at 16kHz
- **FreeVC-s**: Better preservation of source speech content, recommended for maintaining clarity
- **FreeVC (24kHz)**: Higher quality output with better audio fidelity


### Tips for Better Voice Conversion

1. **Use longer reference samples**: 5-10 seconds of clean speech works best
2. **Try multiple reference samples**: Use the secondary reference input for more robust voice profiles
3. **Adjust voice mimicry settings**:
   - Increase voice_match_strength (0.6-0.8) for stronger character matching
   - Use neutralize_source (0.3-0.5) to reduce source voice influence
   - Add presence_boost (0.3-0.5) for more "in the room" sound

## Known Issues and Troubleshooting

1. **File Not Found Errors**:
   - Ensure all checkpoint files are in the correct directory
   - Verify file names match exactly (case-sensitive)

2. **CUDA Out of Memory**:
   - Try processing shorter audio clips
   - Use CPU if GPU memory is insufficient
   - Lower diffusion steps for diffusion-based models

3. **Audio Quality Issues**:
   - Try different models - each has strengths for different source/target voices
   - For diffusion models, lower the noise coefficient if there's static
   - Increase clarity_enhancement for better intelligibility

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
