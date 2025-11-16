import os
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer, DataCollatorCTCWithPadding
import evaluate
from datasets import load_from_disk
import torch

# Load preprocessed dataset
dataset = load_from_disk("processed_data/telugu")
train_dataset = dataset["train"]
eval_dataset = dataset["valid"]

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained(".", from_pt=True)  # loads feature_extractor + tokenizer from current dir (vocab.json)
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-300m")
# Resize model output to vocab size and set pad token
model.resize_token_embeddings(len(processor.tokenizer))
model.config.pad_token_id = processor.tokenizer.pad_token_id

# Data collator for padding
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# Metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    # Decode prediction to strings
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
    # Decode labels to strings (replace -100 with pad token id)
    label_ids = pred.label_ids
    label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

# Training arguments
training_args = TrainingArguments(
    output_dir="wav2vec2_telugu",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    num_train_epochs=10,
    learning_rate=3e-4,
    warmup_steps=500,
    fp16=True,
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,  # pad using feature extractor
    compute_metrics=compute_metrics
)

# Train and save
trainer.train()
trainer.save_model("wav2vec2_telugu")
processor.save_pretrained("wav2vec2_telugu")
print("Training complete. Model saved to 'wav2vec2_telugu/'.")
