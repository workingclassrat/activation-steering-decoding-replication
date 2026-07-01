# activation-steering-decoding-replication
A replication of "Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention" (https://doi.org/10.18653/v1/2025.acl-long.634)

Activation Steering Decoding (ASD) mitigates hallucinations in Large Vision-Language Models (LVLMs) by computing steering vectors from the difference between factual and hallucinated activations and applying these vectors to shift the model's hidden states toward the ground truth.

## Results

| Model        | Accuracy | F1 Score |
|--------------|----------|----------|
| Base         | 0.8456   | 0.8218   |
| Base (paper) | 0.8513   | 0.8603   |
| ASD          | 0.8394   | 0.8126   |
| ASD (paper)  | 0.8801   | 0.8787   |

## Requirements
- Python 3.12
- 15GB GPU memory for LLaVA-1.5-7b

## Setup
1. Clone this repo

2. Install dependencies: pip install -r requirements.txt

4. Download required data:
- MS COCO 2014 train images and annotations: https://cocodataset.org/#download
- MS COCO 2014 val images: https://cocodataset.org/#download
- Visual Contrastive Decoding (VCD) repository: https://github.com/DAMO-NLP-SG/VCD
- LLaVA-1.5-7b (downloaded automatically from HuggingFace on first run)

4. Set paths in `config.py`

## Usage
1. `calibrate.py` (generates steering vectors from the MS COCO train images)
2. `evaluate_base_model.py` (runs LLaVA without steering on POPE benchmark)
3. `evaluate_asd_model.py` (runs LLaVA with ASD on POPE benchmark)

## Notes
I only evaluated the effectiveness of ASD compared to LLaVA1.5-7B on the MS COCO dataset using POPE benchmark (a combination of random, popular, and adversarial). Due to limited GPU RAM, I only ran calibration on 300 images, while the paper used 500, which could have limited the number of tokens and diversity of the samples, reducing the model’s ability to distinguish between factual and hallucinated activations. I also only used 100 evaluation images and likely used different generation settings as the paper did not specify them. Additionally, while I tried setting lambda and alpha to different values to try optimizing my results, I did not conduct a grid search like the paper.
