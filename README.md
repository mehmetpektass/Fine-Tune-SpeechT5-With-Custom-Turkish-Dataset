# 🗣️ Fine-Tuned SpeechT5 for Turkish TTS

## Description  
This project demonstrates how to fine-tune Microsoft's SpeechT5 Text-to-Speech model on a custom Turkish dataset. It leverages the Hugging Face `transformers` library, `librosa` for audio processing, and `SpeechBrain` for generating speaker embeddings (x-vectors).

The pipeline includes a custom text normalization stage that handles Turkish character transliteration and converts numerical digits into their Turkish word equivalents (e.g., "21" -> "yirmi bir") to ensure high-quality speech synthesis.

<br>

## 💻 Run on Google Colab
You can access the full source code, including dataset extraction, training loop, and inference, directly in your browser without any local setup.

[Click the badge](https://colab.research.google.com/drive/1nbom8lpWXoyA4xF67XsbNVPembUvpQkS) below to open the notebook:

<br>

## 🚀 Demo & Weights
The model weights and processor are pushed to the Hugging Face Hub after training. You can download them directly or use the inference script below.

👉 [Download Model Weights from Hugging Face Hub](https://huggingface.co/mehmetPektas/v2.speecht5_finetuned_tts_tr)

<br>

## 🧠 Model Architecture & Pipeline
- **Base Model:** `microsoft/speecht5_tts`

- **Vocoder:** `microsoft/speecht5_hifigan`

- **Speaker Encoder:** `speechbrain/spkrec-xvect-voxceleb`

- **Audio Config:** Mono, 16kHz sampling rate

- **Text Processing:** - Automated number-to-word conversion (Turkish)

- **Character replacement:** (e.g., `ç` -> `ch`, `ş` -> `sh`) due to tokenizer vocabulary limitations.

<br>

## 📉  Training Configuration and The Results
The model was trained using `Seq2SeqTrainer` with the following configurations:

 - ***Max Steps:*** 800

- ***Batch Size:*** 4 (per device)

- ***Gradient Accumulation:*** 8

- ***Learning Rate:*** 5e-4

- ***Optimizer:*** `AdamW`

| Step | Training Loss | Validation Loss |
|:----:|:-------------:|:---------------:|
| 200  | 0.7671        |0.7421           |
| 400  | 0.6787	       |0.6537           |
| 600  | 0.6045        |0.6150           |
| 800  | 0.5662        |0.5704           |


<br>

## 👩🏻‍💻 Installation & Usage

* **1.Clone the repository:**
```
git clone https://github.com/mehmetpektass/Fine-Tune-SpeechT5-With-Custom-Turkish-Dataset.git
cd SpeechT5_Turkish_Fine-Tuning_TTS
```
<br>

* **2.Install Requirements:**
```
pip install torch transformers datasets librosa soundfile speechbrain
```
<br>

* **3. Inference (Generating Text)**
You can generate speech using the fine-tuned model. Create a python script (e.g., `inference.py`) and use the following code.

***Note:*** Since SpeechT5 relies on speaker embeddings, we use a sample embedding or a random x-vector.
```
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier

# 1. Load Models
checkpoint = "your-username/your-model-name" # Replace with your HF hub path
processor = SpeechT5Processor.from_pretrained(checkpoint)
model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# 2. Text Normalization (Crucial Step)
# This function mimics the training preprocessing (replacing chars and numbers)
def normalize_text(text):
    replacements = [("â", "a"), ("ç", "ch"), ("ğ", "gh"), ("ı", "i"), 
                    ("î", "i"), ("ö", "oe"), ("ş", "sh"), ("ü", "ue"), ("û", "u")]
    text = text.lower()
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text

input_text = "Merhaba, bu proje tamamlanmıştır."
normalized_text = normalize_text(input_text)

# 3. Prepare Inputs
inputs = processor(text=normalized_text, return_tensors="pt")

# 4. Load/Generate Speaker Embedding (X-Vector)
spk_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="tmp")
# Dummy embedding for example (normally you would pass a waveform here)
speaker_embeddings = torch.zeros((1, 512)) 

# 5. Generate Speech
print("Generating audio...")
with torch.no_grad():
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

# 6. Save
sf.write("output.wav", speech.numpy(), 16000)
print("Saved to output.wav")
```
<br>

* **4. Training from Scratch:**
```
python3 main.ipynb
```
<br>


## Contribution Guidelines 🚀

##### Pull requests are welcome. If you'd like to contribute, please:
- Fork the repository
- Create a feature branch
- Submit a pull request with a clear description of changes
- Ensure code follows existing style patterns
- Update documentation as needed

