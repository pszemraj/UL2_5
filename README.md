# UL2.5 Data Collator

GPU-ready data collation for encoder-decoder models (T5, FLAN, etc.) implementing and improving the UL2 mixture-of-denoisers training paradigm.

## Features

- **Multiple denoising objectives**: Span corruption, prefix LM, infilling
- **GPU-optimized**: Batch sampling, device-specific caching, minimal CPU-GPU sync
- **HuggingFace compatible**: Works with `Trainer` and `DataLoader`
- **Curriculum learning**: Gradually shift denoiser mixture during training
- **Two implementations**: HF-integrated (`ul2_5_hf.py`) or pure PyTorch (`ul2_5_torch.py`)

---

- [UL2.5 Data Collator](#ul25-data-collator)
  - [Features](#features)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Configuration Presets](#configuration-presets)
  - [Usage with HuggingFace Trainer](#usage-with-huggingface-trainer)
  - [Usage with DataLoader](#usage-with-dataloader)
  - [Curriculum Learning](#curriculum-learning)
  - [Custom Configuration](#custom-configuration)
  - [Denoising Tasks](#denoising-tasks)
  - [API Reference](#api-reference)
  - [Performance Tips](#performance-tips)
  - [Benchmarks](#benchmarks)
  - [Visualizations](#visualizations)
  - [References](#references)

---

## Installation

```bash
pip install torch>=2.0.0 transformers>=4.30.0 sentencepiece pydantic>=2.0.0
```

Or clone and install dependencies:

```bash
git clone <repo-url>
cd UL2_5
pip install -r requirements.txt
```

## Quick Start

```python
from transformers import AutoTokenizer, T5ForConditionalGeneration
from ul2_5_hf import UL25DataCollator, UL25Config

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")

# Add UL2 task prefixes (optional but recommended)
tokenizer.add_special_tokens({
    "additional_special_tokens": ["[R]", "[S]", "[X]", "[I]"]
})
model.resize_token_embeddings(len(tokenizer))

# Create collator
collator = UL25DataCollator(
    tokenizer=tokenizer,
    config=UL25Config.recommended(),
    max_length=512,
    max_labels_length=128,
)

# Use with your data
batch = collator([
    {"input_ids": tokenizer.encode("The quick brown fox jumps over the lazy dog.")},
    {"input_ids": tokenizer.encode("Machine learning is transforming industries.")},
])

print(batch["input_ids"].shape)      # [batch_size, max_enc_len]
print(batch["attention_mask"].shape) # [batch_size, max_enc_len]
print(batch["labels"].shape)         # [batch_size, max_dec_len]
```

## Configuration Presets

| Preset                                     | Description                                         | Use Case               |
| ------------------------------------------ | --------------------------------------------------- | ---------------------- |
| `UL25Config.recommended()`                 | Balanced mixture (30% span, 50% prefix, 20% infill) | General pre-training   |
| `UL25Config.recommended_with_curriculum()` | Starts span-heavy, shifts to prefix-heavy           | Long pre-training runs |
| `UL25Config.ul2_original()`                | Original UL2 paper 7-denoiser mixture               | Reproducing UL2        |
| `UL25Config.t5_standard()`                 | Standard T5 span corruption only                    | T5-style training      |

## Usage with HuggingFace Trainer

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    # ... other args
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # Dataset returning {"input_ids": [...]}
    data_collator=collator,
)

trainer.train()
```

## Usage with DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collator,
    num_workers=4,
    pin_memory=True,
)

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    # ...
```

## Curriculum Learning

Gradually shift the denoiser mixture during training:

```python
from ul2_5_hf import UL25DataCollator, UL25Config

collator = UL25DataCollator(
    tokenizer=tokenizer,
    config=UL25Config.recommended_with_curriculum(),
)

# During training, update progress (0.0 to 1.0)
for epoch in range(num_epochs):
    collator.progress = epoch / num_epochs

    for batch in dataloader:
        # At progress=0: heavier on span corruption
        # At progress=1: heavier on prefix LM
        outputs = model(**batch)
        # ...
```

## Custom Configuration

```python
from ul2_5_hf import UL25Config, DenoiserSpec, Task

config = UL25Config(
    denoisers=[
        # Standard span corruption (T5-style)
        DenoiserSpec(
            task=Task.SPAN,
            mu=3.0,           # Mean span length
            r=0.15,           # 15% corruption rate
            prefix="[S]",     # Task prefix token
        ),
        # Extreme span corruption
        DenoiserSpec(
            task=Task.SPAN,
            mu=32.0,          # Longer spans
            r=0.5,            # 50% corruption
            prefix="[X]",
        ),
        # Prefix LM (causal generation)
        DenoiserSpec(
            task=Task.PREFIX_RANDOM,
            prefix="[R]",
        ),
    ],
    weights=[0.4, 0.3, 0.3],  # Sampling probabilities
)

collator = UL25DataCollator(tokenizer, config)
```

## Denoising Tasks

### Span Corruption (`Task.SPAN`)

Standard T5-style: randomly mask contiguous spans, replace with sentinel tokens.

```
Input:  The quick brown fox jumps over the lazy dog.
Output: The quick <extra_id_0> jumps <extra_id_1> dog.
Target: <extra_id_0> brown fox <extra_id_1> over the lazy
```

### Middle-Heavy Span (`Task.SPAN_MIDDLE`)

Position-biased masking preferring middle tokens (Gaussian weighting).

### Prefix LM (`Task.PREFIX_RANDOM/SHORT/LONG`)

Split sequence into prefix (encoder) and suffix (decoder target).

- `PREFIX_RANDOM`: Random split point (20-80%)
- `PREFIX_SHORT`: Short prefix, long target (generation-focused)
- `PREFIX_LONG`: Long prefix, short target (QA-focused)

### Infilling (`Task.INFILLING`)

Mask a contiguous middle chunk, provide bidirectional context.

## API Reference

### `UL25DataCollator`

```python
UL25DataCollator(
    tokenizer,                    # HuggingFace tokenizer with extra_id tokens
    config=None,                  # UL25Config (default: recommended())
    max_length=512,               # Max encoder sequence length
    max_labels_length=128,        # Max decoder sequence length
    pad_to_multiple_of=None,      # Pad to multiple (e.g., 8 for tensor cores)
    return_tensors="pt",          # Output format
)
```

**Properties:**

- `progress`: Float 0.0-1.0 for curriculum learning

**Returns:** `{"input_ids", "attention_mask", "labels"}`

### `UL25Config`

```python
UL25Config(
    denoisers: List[DenoiserSpec],  # List of denoiser specifications
    weights: List[float],           # Sampling probabilities (must sum to 1)
    curriculum_start: List[float],  # Weights at progress=0 (optional)
    curriculum_end: List[float],    # Weights at progress=1 (optional)
)
```

### `DenoiserSpec`

```python
DenoiserSpec(
    task: Task,           # Task type (SPAN, PREFIX_RANDOM, etc.)
    mu: float = 3.0,      # Mean span length (for SPAN tasks)
    r: float = 0.15,      # Noise density / corruption rate
    max_spans: int = 512, # Maximum number of spans
    prefix: str = "",     # Task prefix token ("[S]", "[R]", etc.)
    variable_r: bool = False,  # Sample r uniformly from r_bounds
    r_bounds: Tuple[float, float] = (0.0, 1.0),  # Range for variable_r
)
```

## Performance Tips

1. **Use `pad_to_multiple_of=8`** for tensor core alignment
2. **Pin memory** in DataLoader: `pin_memory=True`
3. **GPU tensors**: Pass input_ids as CUDA tensors for GPU-side processing
4. **Batch size**: Larger batches amortize collation overhead

## Benchmarks

Run the benchmark suite:

```bash
python benchmark_ul2_5.py
```

Sample output (RTX 4070 Laptop):

| Config       | Batch | SeqLen | Tokens/s |
| ------------ | ----- | ------ | -------- |
| recommended  | 32    | 512    | ~380k    |
| ul2_original | 32    | 512    | ~230k    |
| t5_standard  | 32    | 512    | ~240k    |

## Visualizations

### Mask Distributions

![Mask Distributions](assets/mask_distributions_torch.png)

### Mask Examples

![Mask Examples](assets/mask_examples_torch.png)

---

## References

- [UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131)
- [T5: Exploring Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
