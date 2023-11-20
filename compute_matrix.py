import os
from PIL import Image
import evaluate
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import numpy as np
import nltk

# Assuming your model is loaded as 'model', replace 'your_model_name' with the actual model name
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load the Rouge metric
metric = evaluate.load("rouge")

# Set ignore_pad_token_for_loss based on your training setup
ignore_pad_token_for_loss = True

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # RougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # Compute Rouge metrics
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Round and scale the metrics to percentage
    result = {k: round(v * 100, 4) for k, v in result.items()}

    # Additional metric for generation length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    return result

# Save predictions to a text file
output_file_path = 'predictions.txt'

# Read predictions from the text file
with open(output_file_path, 'r') as file:
    lines = file.readlines()

# Extract image names and predictions
image_predictions = {}
for line in lines:
    parts = line.strip().split(':')
    if len(parts) == 2:
        image_file = parts[0].strip()
        prediction = parts[1].strip()
        image_predictions[image_file] = prediction

#ground truth text file
ground_file_path = 'captions.txt'

with open(ground_file_path, 'r') as file:
    lines = file.readlines()

# Extract image names and predictions
ground_truth_data = {}
for line in lines:
    parts = line.strip().split(',')
    if len(parts) == 2:
        image_file = parts[0].strip()
        prediction = parts[1].strip()
        ground_truth_data[image_file] = prediction

# Convert ground truth and predictions to the format expected by compute_metrics
eval_preds = (tokenizer.batch_encode_plus(list(image_predictions.values()), return_tensors="pt")['input_ids'],
              tokenizer.batch_encode_plus(list(ground_truth_data.values()), return_tensors="pt")['input_ids'])

# Compute metrics
accuracy_metrics = compute_metrics(eval_preds)

# Print or use the accuracy_metrics as needed
print("Accuracy Metrics:", accuracy_metrics)
