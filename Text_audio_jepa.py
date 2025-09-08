# -*- coding: utf-8 -*-
"""This Code is Based on a Jupyter notebook made in Google Colab by KIRAN UDAYAKUMAR.
   For MSc Data Science Project EMATM0047 at University of Bristol.
   The notebook can be found at https://colab.research.google.com/drive/1pKpuH_7hHA-FPX8wK0d3DU8k6_0u7blc?usp=sharing
"""

# !pip install x_transformers

"""## Imports"""

from typing import *
from pathlib import Path
import torch
import torch.nn as nn

"""## Download Datasets

Downloading Dataset and assign it into ds
"""

from datasets import load_dataset, Features, Value

ds = load_dataset("ashraq/esc50", split="train")

# Define new column type to override Audio decoding
raw_audio_features = Features({
    "bytes": Value("binary"),
    "path": Value("string"),
})

# Apply casting
ds = ds.cast_column("audio", raw_audio_features)
ds

# checking a random element in ds
ds[10]

"""Keeping necessary columns in ds and removing all other columns"""

# Keep only necessary columns
ds = ds.remove_columns([
    c for c in ds.column_names
    if c not in ["filename", "category", "audio"]
])

"""Format the category values

Replacing '_' with ' '
"""

def format_category(sample):
  sample["category"] = sample["category"].replace("_", " ")
  return sample

ds = ds.map(format_category)

# Examining column names
print(ds.column_names)

"""Checking the sampling rate of audio in ds"""

import torchaudio

i = 10 # Example index
sample = ds[i]
waveform, sample_rate = torchaudio.load(sample["audio"]["bytes"])
print(f"{waveform.shape=}, {sample_rate=}")

"""Resampling function

"""

# Function to resample audio and apply to dataset
def resample_audio_column(sample):
    waveform, original_freq = torchaudio.load(sample["audio"]["bytes"])
    target_freq: int = 24000
    if original_freq == target_freq:    # no resampling

        resampled_waveform = waveform
    else:
        resampler = torchaudio.transforms.Resample(
            orig_freq= original_freq,
            new_freq= target_freq
        )
        resampled_waveform = resampler(waveform)
    sample["audio_24k"] = {"bytes": None, "path": None, "array": resampled_waveform.squeeze().numpy(), "sampling_rate": target_freq}
    return sample

# Applying the resampling functio to all elements of ds
ds = ds.map(resample_audio_column)

print(ds.column_names)

# Replace the original audio column with the resampled one in ds
ds = ds.remove_columns(["audio"])
ds = ds.rename_column("audio_24k", "audio")
print(ds.column_names)

i = 0 # You can change this index to access different elements
print(ds[i]["audio"]["sampling_rate"])

"""Playing a random audio file to make sure dataset was downloaded correctly"""

from IPython.display import Audio

# Select the first sample as an example
sample_to_play = ds[200]

# Extract the audio array and sampling rate from the 'audio' column
audio_array = sample_to_play["audio"]["array"]
sampling_rate = sample_to_play["audio"]["sampling_rate"]
category = sample_to_play["category"]

print(f"Playing sample from category: {category}")

# Play the audio
Audio(data=audio_array, rate=sampling_rate)

"""Splitting the dataset into train and test."""

splits = ds.train_test_split(test_size=0.1)
splits

import torchaudio

sample: dict = splits["train"][0]
sample.keys()

"""Downloading pretrained Encodec model"""

from transformers import EncodecModel, AutoProcessor

AUDIO_MODEL: str = "facebook/encodec_24khz" # @param {type:"string"}

audio_model = EncodecModel.from_pretrained(AUDIO_MODEL)
audio_processor = AutoProcessor.from_pretrained(AUDIO_MODEL)

"""Calculating the fps and number of frames in the audio"""

from operator import mul
from functools import reduce

fps: float = audio_model.config.sampling_rate / reduce(mul, audio_model.config.upsampling_ratios, 1)
print(f"{fps=}")

audio_length_s: int = 5

num_audio_frames: int = int(fps * audio_length_s)
print(f"{num_audio_frames=}")

"""Downloading the text encoder"""

# @title Text Encoder

from sentence_transformers import SentenceTransformer

text_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
text_encoder

import torchaudio

sample: dict = splits["train"][5]
sample.keys()

print(sample["category"])

"""Generating text embeddings from 'category'"""

text_embeddings: torch.Tensor = text_encoder.encode(
    [category],
    convert_to_tensor=True,
    normalize_embeddings=True
).clone()

text_embeddings.shape

"""Preparing the text embeddings and projecting it to the correct shape, to be used as input for JEPA_base"""

import torch.nn as nn

text_embed_dim: int = text_encoder.get_sentence_embedding_dimension()
print(f"{text_embed_dim=}")

num_audio_channels: int = 2 # Based on Encodec model
print(f"{num_audio_channels=}")
print(f"{num_audio_frames=}") # Defined in a previous cell

# Define a linear layer to project text embeddings
proj_text_to_audio = nn.Linear(
    text_embed_dim,
    num_audio_channels * num_audio_frames
)

# Apply the linear projection and reshape to match audio embeddings
text_embeddings_proj = proj_text_to_audio(
    text_embeddings
).view(-1, num_audio_channels, num_audio_frames)

print(f"{text_embeddings_proj.shape=}")

"""Preparing the raw audio into specific format as Enoced and JEPA model expects"""

# Access the audio array from the sample
audio_array = sample["audio"]["array"]

# Process the audio with the audio_processor
processed_audio = audio_processor(
    raw_audio=audio_array,
    sampling_rate=audio_processor.sampling_rate,
    return_tensors="pt"
)

# Print the shape of the processed audio input values
print(processed_audio["input_values"].shape)

"""Function to format category into tensor format"""

def process_category(category: str, text_encoder, proj_text_to_audio, num_audio_channels: int, num_audio_frames: int) -> torch.Tensor:

    # Get text embedding (encode a list containing the single category)
    text_embeddings: torch.Tensor = text_encoder.encode(
        [category],
        convert_to_tensor=True,
        normalize_embeddings=True
    ).clone()

    # Apply linear projection and reshape
    text_embeddings_proj = proj_text_to_audio(
        text_embeddings
    ).view(-1, num_audio_channels, num_audio_frames)

    return text_embeddings_proj

"""testing the function"""

# Get a sample category from the dataset (e.g., the first one)
sample_category = ds[0]["category"]
print(f"Sample category: {sample_category}")

# Test the process_category function
projected_embedding = process_category(
    sample_category,
    text_encoder,
    proj_text_to_audio,
    num_audio_channels,
    num_audio_frames
)

# Print the shape of the output
print(f"Shape of projected embedding: {projected_embedding.shape}")

"""Making tensors of audio and text and storing it in a dictionary as training and test splits"""

processed_audio_data = {}

# Iterate through both train and test splits
for split_name, dataset in splits.items():
    processed_audio_list = []
    print(f"Processing {split_name} split...")
    for i, sample in enumerate(dataset):
        try:
            # Access the audio array from the sample
            audio_array = sample["audio"]["array"]
            category = sample["category"] # Get the category

            # Process the audio with the audio_processor
            processed_audio = audio_processor(
                raw_audio=audio_array,
                sampling_rate=audio_processor.sampling_rate,
                return_tensors="pt"
            )

            # Process the category using the process_category function
            processed_category_tensor = process_category(
                category,
                text_encoder,
                proj_text_to_audio,
                num_audio_channels,
                num_audio_frames
            )

            # Include category tensor and original category string with processed audio
            processed_audio['category_tensor'] = processed_category_tensor
            processed_audio['category'] = category # Add original category string
            processed_audio_list.append(processed_audio)

            # Optional: Print shape and category for a few samples to verify
            if i < 5:
                print(f"  Sample {i} processed audio shape: {processed_audio['input_values'].shape}, Category Tensor shape: {processed_audio['category_tensor'].shape}, Category: {processed_audio['category']}")

        except Exception as e:
            print(f"  Could not process sample {i}: {e}")
            # Handle samples that might cause errors if necessary

    processed_audio_data[split_name] = processed_audio_list

print("Finished processing all audio data.")

print(f"Number of splits in processed_audio_data: {len(processed_audio_data)}")
for split_name, data_list in processed_audio_data.items():
    print(f"Number of samples in '{split_name}' split: {len(data_list)}")

""" Inspecting the structure and content of "processed_audio_data"
"""

print(processed_audio_data.keys())
for key, value in processed_audio_data.items():
    print(f"Key: {key}, Type of value: {type(value)}, Number of items: {len(value)}")
    if len(value) > 0:
        print(f"Type of first item in list: {type(value[0])}")
        if isinstance(value[0], dict):
            print(f"Keys of first item: {value[0].keys()}")
            for sub_key, sub_value in value[0].items():
                print(f"  Sub-key: {sub_key}, Type of sub-value: {type(sub_value)}, Shape of sub-value (if tensor): {sub_value.shape if hasattr(sub_value, 'shape') else 'N/A'}")
    print("-" * 20)

"""JEPA Base"""

import torch
import torch.nn as nn
from x_transformers import Encoder
from typing import Optional, Tuple, Literal, Any, Union


class Predictor(nn.Module):
    """
    Predictor module built on a Transformer Encoder.
    It takes context embeddings (known inputs) and target masks (placeholders for predictions),
    processes them together through a transformer, and outputs predictions only for the target positions.
    """
    def __init__(
        self,
        embed_dim: int, # Dimension of the input embeddings
        num_heads: int, # Number of attention heads in the transformer
        depth: int, # Number of transformer layers
        layer_dropout: float = 0.0, # Dropout rate for the transformer layers
        predictor_embed_dim: Optional[int] = None, # dimension for the predictor's internal embedding
    ):
        super().__init__()
        # Initialize the transformer-based decoder (using x_transformers Encoder)
        self.decoder = Encoder(
            dim=predictor_embed_dim if predictor_embed_dim else embed_dim, # Transformer dimension
            depth=depth,
            heads=num_heads,
            layer_dropout=layer_dropout
        )

        # linear projection to the predictor's internal embedding dimension
        self.predictor_embed = (
            nn.Linear(embed_dim, predictor_embed_dim, bias=True)
            if predictor_embed_dim
            else nn.Identity() # Use Identity if no projection is needed
        )

        # Normalization layer before the final projection
        self.predictor_norm = (
            nn.LayerNorm(predictor_embed_dim) if predictor_embed_dim else nn.Identity()
        )
        # linear projection back to the input embedding dimension
        self.predictor_proj = (
            nn.Linear(predictor_embed_dim, embed_dim, bias=True)
            if predictor_embed_dim
            else nn.Identity()
        )

    def forward(
        self, context_encoding: torch.Tensor, target_masks: torch.Tensor
    ) -> torch.Tensor:

        # Concatenate the context encoding and the target masks
        x = torch.cat(
            (context_encoding, target_masks), dim=1
        )

        # Map context tokens to the predictor dimension
        x = self.predictor_embed(x)

        # Pass the concatenated tensor through the transformer decoder
        x = self.decoder(x)

        # Normalise and project predictor outputs back to the input dimension
        x = self.predictor_proj(
            self.predictor_norm(x)
        )

        # Return the output corresponding to target tokens

        # Predictions
        prediction = x[
            :, -target_masks.shape[1] :, :  # Slice to get only the target positions
        ]
        return prediction


class JEPA_base(nn.Module):

    def __init__(
        self,
        audio_encoder: nn.Module, # The pre-trained audio encoder (teacher)
        text_encoder: nn.Module, # The pre-trained text encoder (student)
        decoder_depth: int, # Depth of the predictor transformer decoder
        num_heads: int, # Number of attention heads for the predictor
        num_audio_channels: int, # Number of channels in the audio embeddings
        num_audio_frames: int, # Number of frames in the audio embeddings after processing
        predictor_embed_dim: Optional[int] = None, # internal dimension for the predictor
        post_enc_norm: bool = False, # Whether to apply layer norm after encoders
        mode: Literal["test", "train"] = "train", # Model mode: 'train' or 'test'

        # Masking parameters
        context_ratio_range: Tuple[float, float] = (0.85, 0.95), # Range for context block ratio
        target_mask_range: Tuple[float, float] = (0.15, 0.25), # Range for target mask ratio
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_audio_channels = num_audio_channels
        self.num_audio_frames = num_audio_frames
        self.mode = mode.lower()
        self.context_ratio_range = context_ratio_range
        self.target_mask_range = target_mask_range
        self.predictor_embed_dim = predictor_embed_dim

        self.embed_dim = self.num_audio_frames

        # Dimension of text embeddings from the text encoder
        self.text_embed_dim = text_encoder.get_sentence_embedding_dimension()


        self.mask_token = nn.Parameter(
            torch.randn(1, self.num_audio_channels, self.num_audio_frames)
        )
        nn.init.trunc_normal_(self.mask_token, 0.02)

        # Post-encoder normalization
        self.post_enc_norm = post_enc_norm
        self.post_enc_norm_jepa = (
            nn.LayerNorm(self.embed_dim) if self.post_enc_norm else nn.Identity()
        )

        # Assign the audio encoder (teacher)
        self.audio_encoder = audio_encoder
        # Freeze audio encoder parameters as it acts as the fixed teacher
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

        # Initialize the Predictor module
        self.predictor = Predictor(
            embed_dim=self.num_audio_frames,
            num_heads=self.num_heads,
            depth=decoder_depth,
            predictor_embed_dim=self.predictor_embed_dim,
        )

        # Assign the text encoder (student)
        self.text_encoder = text_encoder


        # Project text embeddings to the shape of audio embeddings
        self.proj_text_to_audio = nn.Linear(
            self.text_embed_dim,
            self.num_audio_channels * self.num_audio_frames
        )


    def forward_base(
        self,
        *,
        audio: dict[str, torch.Tensor], # Processed audio data from audio_processor
        text: list[str], # List of category strings
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, dict], torch.Tensor]:


        test_mode: bool = self.mode == "test"

        # Encode the text strings into embeddings using the text encoder (student)
        text_embeddings: torch.Tensor = self.text_encoder.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).clone()

        # Project and reshape text embeddings to match the audio embedding shape
        text_embeddings = self.proj_text_to_audio(
            text_embeddings
        ).view(-1, self.num_audio_channels, self.num_audio_frames)


        # In test mode, return the projected text embeddings
        if test_mode:
            return text_embeddings

        # In train mode, encode audio using the frozen audio encoder (teacher)
        with torch.no_grad():
            # Prepare audio inputs for the audio encoder
            audio_input_values = audio["input_values"].squeeze(1)
            audio_padding_mask = audio["padding_mask"].float() # Cast padding mask to float

            # Check and add dimension to input values if needed
            if audio_input_values.ndim == 2:
                audio_input_values = audio_input_values.unsqueeze(1)

             # Check and add dimension to padding mask if needed
            if audio_padding_mask.ndim == 2:
                audio_padding_mask = audio_padding_mask.unsqueeze(1)

            # Add shape check here
            if audio_input_values.ndim == 3 and audio_padding_mask.ndim == 3:

                # Encode audio
                audio_encoder_output = self.audio_encoder.encode(
                    audio_input_values,
                    audio_padding_mask
                )
            else:
                print(f"audio_input_values shape: {audio_input_values.shape}")
                print(f"audio_padding_mask shape: {audio_padding_mask.shape}")
                raise ValueError("KR!Shapes are not as expected. Stopping execution.")


            # Get the continuous audio embeddings (teacher targets)
            audio_embeddings: torch.Tensor = audio_encoder_output.to_tuple()[0]
            # Assuming the output is [B, C, T_frames], squeeze if needed
            if audio_embeddings.ndim == 4:
                audio_embeddings = audio_embeddings.squeeze(0)


        # Prepare target masks for the predictor input
        target_masks: torch.Tensor = self.mask_token.repeat(audio_embeddings.shape[0], 1, 1)

        predictions: torch.Tensor = self.predictor(
            text_embeddings, # Context: projected text embeddings
            target_masks # Target: mask tokens
        )

        # Slice audio_embeddings (teacher targets) to match the sequence length of predictions
        audio_embeddings_sliced = audio_embeddings[:, :, :predictions.shape[-1]]


        # Prepare extra information to be returned (useful for loss calculation and debugging)
        extras = {
          "audio_embeddings": audio_embeddings,                       # Original audio embeddings from teacher [B, 2, T_audio]
          "audio_embeddings_sliced": audio_embeddings_sliced,         # Sliced audio embeddings for target [B, 2, num_audio_frames]
          "target_masks": target_masks                               # Mask tokens used as predictor input [B, 2, num_audio_frames]
        }

        # Return predictions, corresponding targets, and extras for loss calculation
        return (
            predictions,
            audio_embeddings_sliced,
            extras
        )

"""Initiating JEPA"""

# Instantiate the JEPA_base model
jb = JEPA_base(
    audio_encoder=audio_model,
    text_encoder=text_encoder,
    decoder_depth=1,
    num_heads=1,
    num_audio_channels=num_audio_channels,
    num_audio_frames=num_audio_frames,
    predictor_embed_dim=256,
)

"""testing JEPA_base"""

# Get a sample batch from processed_audio_data
sample_batch = processed_audio_data["train"][0]

# Prepare the data for the forward pass

audio_input = {
    "input_values": sample_batch["input_values"],
    "padding_mask": sample_batch["padding_mask"]
}
text_input = [sample_batch["category"]] # forward_base expects a list


# Add a batch dimension to the tensors as processed_audio_data items are single samples
audio_input_batched = {k: v.unsqueeze(0) if v.ndim == 3 else v for k, v in audio_input.items()}


# Ensure text_input is a list even if batch size is 1
text_input_batched = text_input # Already a list of strings

# Perform the forward pass using the jb model
predictions, targets, extras = jb.forward_base(
    audio=audio_input_batched,
    text=text_input_batched,
)

print("Forward pass successful!")
print(f"Predictions shape: {predictions.shape}")
print(f"Targets shape: {targets.shape}")
print(f"Extras keys: {extras.keys()}")

"""Creating a dataloader to access each elements in split

"""

from torch.utils.data import Dataset, DataLoader
import torch

class ProcessedAudioDataset(Dataset):
    """
    A simple Dataset to wrap the processed audio data from processed_audio_data.
    """
    def __init__(self, processed_data_list):
        self.processed_data_list = processed_data_list

    def __len__(self):
        return len(self.processed_data_list)

    def __getitem__(self, idx):
        # Return the dictionary for the specific sample
        return self.processed_data_list[idx]

def custom_collate_fn(batch):
    """
    Custom collate function to batch processed audio data and categories.
    """
    input_values_list = []
    padding_mask_list = []
    category_tensor_list = []
    category_list = []

    for sample in batch:
        input_values_list.append(sample["input_values"])
        padding_mask_list.append(sample["padding_mask"])
        category_tensor_list.append(sample["category_tensor"])
        category_list.append(sample["category"])

    # Stack the tensors along the batch dimension
    input_values = torch.cat(input_values_list, dim=0)
    padding_mask = torch.cat(padding_mask_list, dim=0)
    category_tensors = torch.cat(category_tensor_list, dim=0)

    return {
        "audio": {
            "input_values": input_values,
            "padding_mask": padding_mask,
        },
        "category_tensors": category_tensors,
        "categories": category_list, # Keep categories as a list of strings
    }


# Create instances of the dataset for train and test splits
train_dataset = ProcessedAudioDataset(processed_audio_data["train"])
test_dataset = ProcessedAudioDataset(processed_audio_data["test"])

# Create dataloaders using the custom_collate_fn
train_dataloader = DataLoader(
    train_dataset,
    batch_size=12,
    shuffle=True,
    collate_fn=custom_collate_fn,
    drop_last=True # Drop the last incomplete batch if the dataset size is not divisible by batch size
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=12,
    shuffle=False, # No need to shuffle test data
    collate_fn=custom_collate_fn,
    drop_last=False # Keep the last incomplete batch in test set
)

print("Dataloaders created successfully.")
print(f"Number of batches in train_dataloader: {len(train_dataloader)}")
print(f"Number of batches in test_dataloader: {len(test_dataloader)}")

"""Creating training and testing loops"""

# Create iterators for the dataloaders
train_dataloader_iter = iter(train_dataloader)
test_dataloader_iter = iter(test_dataloader)

print("Dataloader iterators created successfully.")

"""Checking if Dataloaders are working fine"""

# Get and display the next batch from the train dataloader iterator
print("Next batch from train_dataloader_iter:")
train_batch = next(train_dataloader_iter)
# Display relevant parts of the batch (e.g., shapes of tensors and first few categories)
print("  Audio input values shape:", train_batch["audio"]["input_values"].shape)
print("  Audio padding mask shape:", train_batch["audio"]["padding_mask"].shape)
print("  Category tensors shape:", train_batch["category_tensors"].shape)
print("  Categories (first few):", train_batch["categories"][:5])
print("-" * 30)


# Get and display the next batch from the test dataloader iterator
print("Next batch from test_dataloader_iter:")
test_batch = next(test_dataloader_iter)
# Display relevant parts of the batch
print("  Audio input values shape:", test_batch["audio"]["input_values"].shape)
print("  Audio padding mask shape:", test_batch["audio"]["padding_mask"].shape)
print("  Category tensors shape:", test_batch["category_tensors"].shape)
print("  Categories (first few):", test_batch["categories"][:5])

"""training

Defining optimizer for adjusting model parameters
"""

# Optimizer
from torch.optim import AdamW

optimizer = AdamW(jb.parameters(), lr=1e-4) # learning rate

"""Defining Loss Functon"""

# Loss Function
from torch.nn import MSELoss
criterion = torch.nn.MSELoss(reduction="none")

"""Defining train_step()

 Core unit of  training loop, handling all the computations and parameter updates for a single batch of data to train JEPA_base mode
"""

# Defining train_step
import torch

def train_step(model, batch, criterion, optimizer, device=None):

    model.train()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Unpack batch ---
    audio = batch["audio"]
    text  = batch.get("categories", batch.get("text"))  # raw text list if available

    # Ensure audio shapes for EnCodec and move to device
    input_values = batch["audio"]["input_values"].float().to(device) # -> [B, C, T]

    if input_values.dim() == 2:
        # [B, T] -> [B, 1, T]
        input_values = input_values.unsqueeze(1)
    elif input_values.dim() == 3 and input_values.shape[1] not in (1, 2) and input_values.shape[-1] in (1, 2):

        input_values = input_values.permute(0, 2, 1)

    padding_mask = batch["audio"].get("padding_mask")


    if padding_mask is None:
        padding_mask = torch.zeros(input_values.shape[0], input_values.shape[-1], dtype=torch.bool, device=device)
    else:

        padding_mask = padding_mask.to(dtype=torch.bool, device=device)
        if padding_mask.dim() == 3:
            if padding_mask.shape[1] == 1:
                padding_mask = padding_mask[:, 0, :]
            elif padding_mask.shape[2] == 1:
                padding_mask = padding_mask[:, :, 0]
            else:
                padding_mask = padding_mask.any(dim=1)  # collapse channel dim


        T = input_values.shape[-1]
        if padding_mask.shape[1] != T:
            if padding_mask.shape[1] > T:
                padding_mask = padding_mask[:, :T]
            else:
                pad = T - padding_mask.shape[1]
                padding_mask = torch.nn.functional.pad(padding_mask, (0, pad), value=False)


    audio = {
        "input_values": input_values,
        "padding_mask": padding_mask,
        }
    # --- Forward ---
    ret = model.forward_base(audio=audio, text=text)

    # Normalize return to (preds, targs, extras)
    if not isinstance(ret, tuple):
        raise ValueError(f"forward_base should return a tuple in train mode. Got: {type(ret)}")

    if len(ret) == 3:
        preds, targs, extras = ret
    elif len(ret) == 2:
        preds, targs = ret
        extras = {}
    else:
        raise ValueError(f"forward_base returned {len(ret)} values; expected 2 or 3.")

    # Ensure preds and targs are on the correct device and float
    preds  = preds.to(device).float()
    targs  = targs.to(device).float()

    # Align shapes (time / frames last)
    if preds.shape != targs.shape:
        min_frames = min(preds.shape[-1], targs.shape[-1])
        preds = preds[..., :min_frames]
        targs = targs[..., :min_frames]
        # Align channel dimension if needed
        if preds.shape[-2] != targs.shape[-2]:
             min_ch = min(preds.shape[-2], targs.shape[-2])
             preds = preds[..., :min_ch, :]
             targs = targs[..., :min_ch, :]

    # --- Loss Calculation ---
    loss_mask = torch.ones_like(preds, dtype=torch.float, device=device)



    loss = (criterion(preds, targs) * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return float(loss.detach().item())

# Get a batch from the train dataloader iterator
batch = next(train_dataloader_iter)

display(batch)

"""Verifing train_step()"""

# Get a batch from the train dataloader iterator
batch = next(train_dataloader_iter)

# Execute a single training step using the train_step function
loss = train_step(model=jb, batch=batch, criterion=criterion, optimizer=optimizer)

print(f"Loss after one training step: {loss:.4f}")

"""verifing Loss calculation logic"""

mask = extras.get("target_masks")
if mask is None:
    mask = torch.ones_like(predictions)

preds   = predictions.float()
targets = targets.float()
mask    = mask.float().to(preds.device)

# criterion must be MSELoss(reduction="none")
loss = (criterion(preds, targets) * mask).sum() / mask.sum().clamp_min(1.0)

print(loss)

"""Training loop to iterate over the epoches and batches"""

import time

epochs = 5  # Define the number of epochs
train_losses = []

print("Starting training...")

for epoch in range(epochs):
    epoch_start_time = time.time()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        # Perform a single training step
        loss = train_step(model=jb, batch=batch, criterion=criterion, optimizer=optimizer)
        total_loss += loss

        # Optional: Print loss periodically
        if (step + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss:.4f}")

    avg_epoch_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_epoch_loss)
    epoch_end_time = time.time()
    print(f"Epoch [{epoch+1}/{epochs}] finished. Average Loss: {avg_epoch_loss:.4f}, Time: {epoch_end_time - epoch_start_time:.2f}s")

print("Training finished.")

"""testing the model

loss graph
"""

import matplotlib.pyplot as plt

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

"""### Interpretation of the Training Loss Graph

The graph above shows the training loss of the `JEPA_base` model over 5 epochs.

*   **Clear Downward Trend:** The prominent feature of the graph is the consistent decrease in the average training loss with each passing epoch.
*   **Model Learning:** This downward trend is a strong indication that the model is actively learning. The optimizer is successfully adjusting the model's trainable parameters to minimize the Mean Squared Error between the predicted audio embeddings and the target audio embeddings.
*   **Stable Training:** The decline appears relatively smooth over these 5 epochs, suggesting a stable training process without major oscillations or convergence issues within this timeframe.
*   **Loss Scale:** The loss values themselves are large, which is expected given the magnitude of the audio embeddings being predicted. The absolute value is less critical than the observed decreasing trend.

Overall, the graph demonstrates that the training of the `JEPA_base` model is progressing effectively, with the model continuously improving its ability to predict audio representations from text inputs.

##Testing

Testing the trained JEPA_base
"""

# Evaluate the model on the test set
jb.eval()  # Set the model to evaluation mode
total_test_loss = 0
jb.mode = 'test'
with torch.no_grad():  # Disable gradient calculation for evaluation
    for step, batch in enumerate(test_dataloader):
        # Prepare the data for the forward pass
        audio_input = batch["audio"]
        text_input = batch["categories"]

        original_mode = jb.mode
        jb.mode = 'train' # Temporarily set to train to get predictions and targets


        # Ensure batch data is on the correct device for the model
        audio_input_on_device = {k: v.to(next(jb.parameters()).device) for k, v in audio_input.items()}


        # Perform forward pass to get predictions and targets
        predictions, targets, extras = jb.forward_base(
            audio=audio_input_on_device,
            text=text_input,
        )

        # Restore original mode
        jb.mode = original_mode


        # Calculate loss

        criterion.to(predictions.device) # Move criterion to device

        # Calculate loss using the valid frame mask
        loss_mask = torch.ones_like(predictions, dtype=torch.float, device=predictions.device)
        loss = (criterion(predictions, targets) * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_dataloader)
print(f"Average Test Loss: {avg_test_loss:.4f}")

"""###Interpretation of the Average Test Loss

The Average Test Loss after training is **333645.6140**.

*   **Comparison to Training Loss:** The average training loss over the epochs ranged approximately from **339,000 to 360,000**, ending around **339,169.5319** in the final epoch. The test loss (`333645.6140`) is very close to the final training loss.
*   **Interpretation:** A test loss that is close to the training loss is a good sign. It indicates that the model has **generalized well** to unseen data and is not significantly **overfitting** to the training set. The model's performance on data it hasn't trained on is comparable to its performance on the training data.

This result suggests that the model has learned meaningful representations that are applicable beyond the specific examples it saw during training.
"""



"""## Testing with real-time user input

This cell is designed to take text input directly from the user and process it into the same projected embedding format that JEPA_base model expects as input
"""

# Get text input from the user (you can change the prompt as needed)
user_text_input = input("Enter the text for audio prediction: ")

# Ensure the model is in evaluation mode
jb.eval()

# Process the text input
with torch.no_grad():

    # Encode the text input using the text encoder
    text_embeddings: torch.Tensor = text_encoder.encode(
        [user_text_input], # Encode as a list
        convert_to_tensor=True,
        normalize_embeddings=True
    ).clone()

    # Project text embeddings to the shape of audio embeddings
    # Use the proj_text_to_audio layer from the model
    projected_text_embeddings = jb.proj_text_to_audio(
        text_embeddings
    )

    # Reshape to match the expected audio embedding shape (Batch Size, Channels, Frames)
    # Assuming Batch Size is 1 for a single input
    processed_text_input = projected_text_embeddings.view(1, num_audio_channels, num_audio_frames)


print(f"Original text input: '{user_text_input}'")
print(f"Shape of processed text input for prediction: {processed_text_input.shape}")

"""This cell takes the processed text input generated in the previous cell and uses JEPA_base model's predictor component to generate audio embedding."""

# Ensure the model is in evaluation mode
jb.eval()

with torch.no_grad():

    dummy_target_masks = jb.mask_token.repeat(processed_text_input.shape[0], 1, 1).to(processed_text_input.device)


    # Call the predictor part of the model
    predicted_audio_embeddings = jb.predictor(
        processed_text_input,
        dummy_target_masks
    )

print(f"Shape of predicted audio embeddings from predictor: {predicted_audio_embeddings.shape}")