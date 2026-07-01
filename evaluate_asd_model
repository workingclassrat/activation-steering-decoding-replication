from config import STEERING_VECTORS, EVALUATION_IMAGES, POPE_QUESTIONS, ASD_RESULTS

import json
import torch
from PIL import Image
import os
import gc

STEERING_FILE = STEERING_VECTORS
steering_vectors = torch.load(STEERING_FILE, weights_only=False)

lambda_pos = -0.05
lambda_neg = 0.05
alpha_param = 1.0

IMAGE_DIR = EVALUATION_IMAGES
gt_path = POPE_QUESTIONS
out_path_steered = ASD_RESULTS

def create_steering_hook(steering_sign, lambda_val):
    def hook(module, input, output):
        layer_idx = getattr(module, 'layer_idx', None)

        if layer_idx is not None and layer_idx in steering_vectors:
            hidden_states = output[0]

            v = steering_vectors[layer_idx].to(
                hidden_states.device,
                dtype=hidden_states.dtype
            )

            if v.dim() == 1:
                v = v.unsqueeze(0)

            hidden_states.data[:, -1, :] += (steering_sign * lambda_val * v)

        return output

    return hook


def apply_steering_hooks(steering_sign, lambda_val):
    handles = []
    for i, layer in enumerate(model.model.layers):
        layer.layer_idx = i
        handle = layer.register_forward_hook(create_steering_hook(steering_sign, lambda_val))
        handles.append(handle)
    return handles


def remove_all_hooks(handles):
    for h in handles:
        h.remove()

def run_pope_evaluation():

    with open(gt_path, 'r') as f:
        content = f.read().strip()
        if content.startswith('['):
            pope_questions = json.loads(content)
        else:
            pope_questions = [json.loads(l) for l in content.splitlines()]

    folder_files = {f: os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    valid_tasks = [q for q in pope_questions if q['image'] in folder_files]

    results = []
    debug_samples = []

    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]

    for idx, item in enumerate(valid_tasks):
        img_path = folder_files[item['image']]

        image = Image.open(img_path).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        prompt = f"{DEFAULT_IMAGE_TOKEN}\nUSER: {item['text']} Please answer this question with one word.\nASSISTANT:"
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        for layer in model.model.layers:
            layer._forward_hooks.clear()

        pos_handles = apply_steering_hooks(+1.0, lambda_pos)

        with torch.no_grad():
            out_pos = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=1,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True
            )
            logits_pos = out_pos.scores[0][0]

        remove_all_hooks(pos_handles)

        neg_handles = apply_steering_hooks(-1.0, lambda_neg)

        with torch.no_grad():
            out_neg = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=1,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True
            )
            logits_neg = out_neg.scores[0][0]

        remove_all_hooks(neg_handles)

        final_logits = (1 + alpha_param) * logits_pos - alpha_param * logits_neg

        if torch.isnan(final_logits).any() or torch.isinf(final_logits).any():
            final_logits = logits_pos

        pred_asd = "yes" if final_logits[yes_id] > final_logits[no_id] else "no"

        results.append({
            "question_id": item.get('question_id', idx),
            "prompt": item['text'],
            "text": pred_asd,
            "answer": item.get('label', ''),
            "image": item['image']
        })

        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    os.makedirs(os.path.dirname(out_path_steered), exist_ok=True)
    with open(out_path_steered, 'w') as f:
        for entry in results:
            f.write(json.dumps(entry) + '\n')
