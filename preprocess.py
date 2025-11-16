import os
import re
import json
from datasets import load_dataset, DatasetDict
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor

# Check for Hugging Face token if Kathbath is gated
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("Kathbath dataset is gated. Please set the HF_TOKEN environment variable with your Hugging Face token.")

# Load the Kathbath Telugu dataset
dataset = load_dataset("ai4bharat/Kathbath", "telugu", use_auth_token=hf_token)
print(f"Loaded Kathbath Telugu dataset: {dataset}")

# Build vocabulary from all training transcripts (character-level)
chars = set()
for text in dataset["train"]["text"]:
    text = text.lower().strip()
    # Keep letters and spaces only
    text = re.sub(r"[^a-z ']", "", text)
    chars.update(list(text))
# If space is in chars, replace it with '|' as the word delimiter
if " " in chars:
    chars.remove(" ")
    chars.add("|")
# Sort and index
vocab_dict = {c: i for i, c in enumerate(sorted(chars))}
# Add CTC special tokens
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
# Save vocab.json
with open("vocab.json", "w") as f:
    json.dump(vocab_dict, f)
print(f"Saved vocabulary with {len(vocab_dict)} tokens.")

# Create tokenizer and processor
tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Function to map audio and text to model inputs and labels
def prepare_dataset(batch):
    # Load audio array and sampling rate
    audio = batch["audio"]
    # Extract input values (Wav2Vec2 expects list of floats at 16kHz)
    input_values = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_values"] = input_values
    # Clean and encode transcript
    text = batch["text"].lower().strip()
    text = re.sub(r"[^a-z ']", "", text)
    text = text.replace(" ", "|")  # use '|' as word delimiter
    with processor.as_target_processor():
        batch["labels"] = processor(text).input_ids
    return batch

# Apply preprocessing to train and valid splits
train_prepared = dataset["train"].map(prepare_dataset, remove_columns=dataset["train"].column_names)
valid_prepared = dataset["valid"].map(prepare_dataset, remove_columns=dataset["valid"].column_names)

# Combine and save processed dataset
processed_dataset = DatasetDict({
    "train": train_prepared,
    "valid": valid_prepared
})
processed_dataset.save_to_disk("processed_data/telugu")
print("Saved processed data to 'processed_data/telugu'.")
