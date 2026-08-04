from config import COCO_ANNOTATIONS, CALIBRATION_IMAGES, STEERING_VECTORS

import os
import re
import json
import torch
import time
import gc
import random
from PIL import Image
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize

for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)

ann_file = COCO_ANNOTATIONS
with open(ann_file, 'r') as f:
    coco_data = json.load(f)

img_id_to_objects = defaultdict(set)
cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

for ann in coco_data['annotations']:
    img_id_to_objects[ann['image_id']].add(cat_id_to_name[ann['category_id']])

coco_object_categories = list(cat_id_to_name.values())

del coco_data
gc.collect()

def get_sentence_labels(generated_text, ground_truth_set):
    sentences = sent_tokenize(generated_text.lower())
    sentence_labels = []

    for sent in sentences:
        mentioned_objects = set()
        for obj in coco_object_categories:
            if re.search(r'\b' + re.escape(obj.lower()) + r'\b', sent):
                mentioned_objects.add(obj)

        hallucinated_objects = mentioned_objects - ground_truth_set
        label = 1 if len(hallucinated_objects) > 0 else 0
        sentence_labels.append((sent, label))

    return sentence_labels

num_layers = len(model.model.layers)

coco_train_path = CALIBRATION_IMAGES
image_paths = [os.path.join(coco_train_path, f) for f in os.listdir(coco_train_path) if f.endswith('.jpg')]

NUM_IMAGES = 300
image_paths = image_paths[:NUM_IMAGES]

prompts = [
    f"{DEFAULT_IMAGE_TOKEN}\nPlease describe the image in detail.",
    f"{DEFAULT_IMAGE_TOKEN}\nDescribe this image in detail.",
    f"{DEFAULT_IMAGE_TOKEN}\nProvide a detailed description of this image.",
]

target_samples = 5000

factual_reservoir = [[] for _ in range(num_layers)]
factual_count_total = 0

hallucinated_reservoir = [[] for _ in range(num_layers)]
hallucinated_count_total = 0

captured_activations = []

def extraction_hook(module, input, output):
    hidden_states = output[0] if isinstance(output, tuple) else output
    captured_activations.append(hidden_states.detach())

handles = [layer.register_forward_hook(extraction_hook) for layer in model.model.layers]

start_time = time.time()

for i, img_path in enumerate(image_paths):
    try:
        filename = os.path.basename(img_path)
        match = re.search(r'(\d+)\.', filename)
        image_id = int(match.group(1)) if match else i

        image = Image.open(img_path).convert('RGB')
        image_data = image_processor.preprocess(image, return_tensors='pt')
        image_tensor = image_data['pixel_values'].half().cuda()

        prompt = prompts[i % len(prompts)]
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        captured_activations.clear()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, images=image_tensor, do_sample=True,
                temperature=1.3, max_new_tokens=250,
                top_p=0.95, top_k=50,
                no_repeat_ngram_size=2, use_cache=True
            )

        output_ids_filtered = output_ids[0][output_ids[0] >= 0]
        if output_ids_filtered.max() >= tokenizer.vocab_size:
            output_ids_filtered = output_ids_filtered[output_ids_filtered < tokenizer.vocab_size]
        generated_text = tokenizer.decode(output_ids_filtered, skip_special_tokens=True)

        truth = img_id_to_objects.get(image_id, set())
        sentence_labels = get_sentence_labels(generated_text, truth)

        if len(captured_activations) >= num_layers and len(sentence_labels) > 0:
            for layer_idx in range(num_layers):
                act = captured_activations[layer_idx]
                act_seq = (act[0] if isinstance(act, tuple) else act).squeeze(0).cpu().float()

                current_pos = 0
                for sentence, label in sentence_labels:
                    sent_tokens = tokenizer.encode(sentence, add_special_tokens=False)
                    num_tokens = len(sent_tokens)
                    end_pos = min(current_pos + num_tokens, act_seq.size(0))
                    sentence_token_acts = act_seq[current_pos:end_pos]

                    for token_act in sentence_token_acts:
                        if label == 0:  # factual
                            factual_count_total += 1
                            if len(factual_reservoir[layer_idx]) < target_samples:
                                factual_reservoir[layer_idx].append(token_act.clone())
                            else:
                                j = random.randint(0, factual_count_total - 1)
                                if j < target_samples:
                                    factual_reservoir[layer_idx][j] = token_act.clone()
                        else:
                            hallucinated_count_total += 1
                            if len(hallucinated_reservoir[layer_idx]) < target_samples:
                                hallucinated_reservoir[layer_idx].append(token_act.clone())
                            else:
                                j = random.randint(0, hallucinated_count_total - 1)
                                if j < target_samples:
                                    hallucinated_reservoir[layer_idx][j] = token_act.clone()

                    current_pos = end_pos

                del act_seq

        captured_activations.clear()

        if (i + 1) % 25 == 0:
            fact_stored = len(factual_reservoir[0])
            hall_stored = len(hallucinated_reservoir[0])
            print(f"[{i+1}/{NUM_IMAGES}] Total seen: F={factual_count_total}, H={hallucinated_count_total} | "
                  f"Stored: F={fact_stored}, H={hall_stored}")
            torch.cuda.empty_cache()
            gc.collect()

        if (i + 1) % 50 == 0:
            del image, image_data, image_tensor, input_ids, output_ids
            gc.collect()

for h in handles:
    h.remove()

target_count = min(len(factual_reservoir[0]), len(hallucinated_reservoir[0]))

steering_vectors = {}
for layer_idx in range(num_layers):
    factual_sample = factual_reservoir[layer_idx][:target_count]
    hallucinated_sample = hallucinated_reservoir[layer_idx][:target_count]

    mean_factual = torch.stack(factual_sample).mean(dim=0)
    mean_hallucinated = torch.stack(hallucinated_sample).mean(dim=0)

    steering_vector = mean_factual - mean_hallucinated

    steering_vectors[layer_idx] = steering_vector

save_path = STEERING_VECTORS
torch.save(steering_vectors, save_path)
