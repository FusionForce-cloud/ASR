import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_from_disk
import evaluate

# Load model and processor
processor = Wav2Vec2Processor.from_pretrained("wav2vec2_telugu")
model = Wav2Vec2ForCTC.from_pretrained("wav2vec2_telugu")

# Load processed validation set
dataset = load_from_disk("processed_data/telugu")
eval_dataset = dataset["valid"]

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

predictions, references = [], []
for ex in eval_dataset:
    input_vals = torch.tensor(ex["input_values"]).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_vals).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    pred_text = processor.decode(pred_ids, skip_special_tokens=True).replace("|", " ")
    label_text = processor.decode(ex["labels"], skip_special_tokens=True).replace("|", " ")
    predictions.append(pred_text.lower())
    references.append(label_text.lower())

wer = wer_metric.compute(predictions=predictions, references=references)
cer = cer_metric.compute(predictions=predictions, references=references)
print(f"WER: {wer:.3f}")
print(f"CER: {cer:.3f}")
