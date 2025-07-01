# Voice Cloning with CSM-1B

This repository contains tools to clone your voice using the Sesame CSM-1B model. It provides two methods for voice cloning:

1. Local execution on your own GPU
2. Cloud execution using Modal

> **Note:** While this solution does capture some voice characteristics and provides a recognizable clone, it's not the best voice cloning solution available. The results are decent but not perfect. If you have ideas on how to improve the cloning quality, feel free to contribute!

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (for local execution)
- Hugging Face account with access to the CSM-1B model
- Hugging Face API token

## Installation

1. Clone this repository:

```bash
git clone https://github.com/isaiahbjork/csm-voice-cloning.git
cd csm-voice-cloning
```

## Jetson Setup (ARM64 / aarch64)

Running CSM-1B locally on a Jetson requires ARM-built wheels for PyTorch and its audio / quantization companions.
Follow the steps below **exactly**—mixing wheel versions or JetPack releases will almost always end in cryptic "illegal instruction" or CUDA mismatch errors.

> **Compatible JetPack**: These wheels were built against **JetPack 6.2 (CUDA 12.6)**.
> If you are still on JetPack 5 or earlier, upgrade first.
> **Python version**: The instructions assume you are using the system Python 3.10 that ships with JetPack 6.2 (the default `/usr/bin/python3`). If you switch to a different Python installation, be sure it is also **3.10** so the `cp310` wheels below can load.

---

### 1 ‒ Install cuSPARSE Lt library

PyTorch requires the cuSPARSE Lt (Light) library which is not included in JetPack 6.2 by default. Install it first:

```bash
# Add NVIDIA's CUDA repository GPG key
wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/3bf863cc.pub | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-cuda-archive-keyring.gpg

# Add NVIDIA's CUDA repository for ARM64
echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/ /" | sudo tee /etc/apt/sources.list.d/cuda-ubuntu2204-sbsa.list

# Update package list and install cuSPARSE Lt
sudo apt update
sudo apt install libcusparselt0 libcusparselt-dev
```

---

### 2 ‒ Install NVIDIA's Jetson-optimised PyTorch

NVIDIA publishes a single wheel that matches each JetPack release. Download or copy it to your Jetson and install with `pip`:

```bash
# Example for JetPack 6.2 ‒ adjust the filename for your release
pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

_(If the file lives elsewhere, just point to the correct path.)_

---

### 3 ‒ Download and install the custom wheels

Download each wheel manually from the release page, then install them:

1. Visit the release page: [https://github.com/onlykit-ar/csm-voice-cloning/releases/tag/jetson-csm](https://github.com/onlykit-ar/csm-voice-cloning/releases/tag/jetson-csm)

2. Download the following wheels by clicking on each link:

   - [torchao-0.9.0+git14cfbc74-cp310-abi3-linux_aarch64.whl](https://github.com/onlykit-ar/csm-voice-cloning/releases/download/jetson-csm/torchao-0.9.0+git14cfbc74-cp310-abi3-linux_aarch64.whl)
   - [torchaudio-2.5.0-cp310-cp310-linux_aarch64.whl](https://github.com/onlykit-ar/csm-voice-cloning/releases/download/jetson-csm/torchaudio-2.5.0-cp310-cp310-linux_aarch64.whl)
   - [torchtune-0.0.0-py3-none-any.whl](https://github.com/onlykit-ar/csm-voice-cloning/releases/download/jetson-csm/torchtune-0.0.0-py3-none-any.whl)

3. Copy the downloaded wheels to your Jetson (if downloading from another machine) and install them:

```bash
# Install torchao (quantization/optimization helpers)
pip install torchao-0.9.0+git14cfbc74-cp310-abi3-linux_aarch64.whl

# Install torchaudio (audio I/O for PyTorch)
pip install torchaudio-2.5.0-cp310-cp310-linux_aarch64.whl

# Install torchtune (needed to load CSM checkpoints & configs)
pip install torchtune-0.0.0-py3-none-any.whl
```

---

### 4 ‒ Install additional dependencies

Install the remaining Python dependencies using the constraints file to preserve your PyTorch installation:

```bash
# Install remaining dependencies with constraints to protect PyTorch versions
pip install -r requirements.txt --constraint constraints.txt
```

> **Important**: Always use the constraints file when installing Python packages on Jetson to prevent downgrading your carefully installed PyTorch ecosystem.

---

### 5 ‒ Verify the installation

Run a quick import test:

```bash
python - <<'PY'
import torch, torchaudio, torchao, torchtune
import silentcipher, moshi
print("✅  PyTorch:", torch.__version__)
print("✅  TorchAudio:", torchaudio.__version__)
print("✅  TorchAO imported OK")
print("✅  TorchTune imported OK")
print("✅  SilentCipher imported OK")
print("✅  Moshi imported OK")
print("✅  CUDA available:", torch.cuda.is_available())
PY
```

If every line prints successfully, your Jetson environment is ready for **Voice Cloning with CSM-1B**.

> **Note**: These wheels are specifically built for Jetson platforms with CUDA 12.6 support. Update the paths above to point to your actual wheel locations.

## Setting Up Your Hugging Face Token

You need to set your Hugging Face token to download the model. You can do this in two ways:

1. Set it as an environment variable:

```bash
export HF_TOKEN="your_hugging_face_token"
```

2. Or directly in the `voice_clone.py` file:

```python
os.environ["HF_TOKEN"] = "your_hugging_face_token"
```

## Accepting the Model on Hugging Face

Before using the model, you need to accept the terms on Hugging Face:

1. Visit the [Sesame CSM-1B model page](https://huggingface.co/sesame/csm-1b)
2. Click on "Access repository" and accept the terms
3. Make sure you're logged in with the same account that your HF_TOKEN belongs to

## Preparing Your Voice Sample

1. Record a clear audio sample of your voice (2-3 minutes is recommended)
2. Save it as an MP3 or WAV file
3. Transcribe the audio using Whisper or another transcription tool to get the exact text

## Running Voice Cloning Locally

1. Edit the `voice_clone.py` file to set your parameters directly in the code:

```python
# Set the path to your voice sample
context_audio_path = "path/to/your/voice/sample.mp3"

# Set the transcription of your voice sample
# You need to use Whisper or another tool to transcribe your audio
context_text = "The exact transcription of your voice sample..."

# Set the text you want to synthesize
text = "Text you want to synthesize with your voice."

# Set the output filename
output_filename = "output.wav"
```

2. Run the script:

```bash
python voice_clone.py
```

## Running Voice Cloning on Modal

Modal provides cloud GPU resources for faster processing:

1. Install Modal:

```bash
pip install modal
```

2. Set up Modal authentication:

```bash
modal token new
```

3. Edit the `modal_voice_cloning.py` file to set your parameters directly in the code:

```python
# Set the path to your voice sample
context_audio_path = "path/to/your/voice/sample.mp3"

# Set the transcription of your voice sample
# You need to use Whisper or another tool to transcribe your audio
context_text = "The exact transcription of your voice sample..."

# Set the text you want to synthesize
text = "Text you want to synthesize with your voice."

# Set the output filename
output_filename = "output.wav"
```

4. Run the Modal script:

```bash
modal run modal_voice_cloning.py
```

## Important Note on Model Sequence Length

If you encounter tensor dimension errors, you may need to adjust the model's maximum sequence length in `models.py`. The default sequence length is 2048, which works for most cases, but if you're using longer audio samples, you might need to increase this value.

Look for the `max_seq_len` parameter in the `llama3_2_1B()` and `llama3_2_100M()` functions in `models.py` and ensure they have the same value:

```python
def llama3_2_1B():
    return llama3_2.llama3_2(
        # other parameters...
        max_seq_len=2048,  # Increase this value if needed
        # other parameters...
    )
```

## Example

Using a 2 minute and 50 second audio sample works fine with the default settings. For longer samples, you may need to adjust the sequence length as mentioned above.

## Troubleshooting

- **Tensor dimension errors**: Adjust the model sequence length as described above
- **CUDA out of memory**: Try reducing the audio sample length or use a GPU with more memory
- **Model download issues**: Ensure you've accepted the model terms on Hugging Face and your token is correct

## License

This project uses the Sesame CSM-1B model, which is subject to its own license terms. Please refer to the [model page](https://huggingface.co/sesame/csm-1b) for details.
