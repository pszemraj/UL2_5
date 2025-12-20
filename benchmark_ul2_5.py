#!/usr/bin/env python3
"""
UL2.5 PyTorch/HF Collator Benchmark Suite
==========================================

Run with:
    python benchmark_ul2_5.py

Requirements:
    pip install torch transformers matplotlib

Tests:
1. Correctness of all masking functions
2. Speed benchmarks (CPU and GPU if available)
3. Real tokenizer integration
4. Visualization of mask distributions
5. Comparison with NumPy baseline
"""

import time
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# Check dependencies
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
except ImportError:
    print("ERROR: PyTorch required. Install with: pip install torch")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    print("WARNING: matplotlib not found. Visualizations will be skipped.")
    HAS_MATPLOTLIB = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    print("WARNING: transformers not found. Real tokenizer tests will be skipped.")
    HAS_TRANSFORMERS = False

# Import our implementations
from ul2_5_hf import (
    UL25DataCollator,
    UL25Config,
    Task,
    DenoiserSpec,
    span_corruption_mask,
    middle_heavy_mask,
    prefix_lm_mask,
    infilling_mask,
    create_sentinel_ids,
    apply_sentinel_mask,
)


# =============================================================================
# TEST UTILITIES
# =============================================================================

class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name: str = "", sync_cuda: bool = True):
        self.name = name
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.elapsed = 0.0
    
    def __enter__(self):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# CORRECTNESS TESTS
# =============================================================================

def test_masking_correctness(device: torch.device):
    """Test all masking functions for correctness."""
    print("\n" + "=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)
    
    seq_len = 100
    n_samples = 1000
    
    # Test 1: Span corruption density
    print("\n[TEST] span_corruption_mask")
    densities = []
    for _ in range(n_samples):
        mask = span_corruption_mask(seq_len, 0.15, 3.0, 512, device)
        densities.append(mask.float().mean().item())
    
    avg_density = sum(densities) / len(densities)
    print(f"  Target density: 0.15, Actual: {avg_density:.4f}")
    assert 0.12 < avg_density < 0.18, f"Density {avg_density} out of range"
    print("  ✓ Passed")
    
    # Test 2: Middle-heavy position bias
    print("\n[TEST] middle_heavy_mask")
    masks = torch.stack([middle_heavy_mask(seq_len, 0.15, device) for _ in range(n_samples)])
    avg_mask = masks.float().mean(dim=0)
    
    middle_density = avg_mask[25:75].mean().item()
    edge_density = (avg_mask[:25].mean().item() + avg_mask[75:].mean().item()) / 2
    print(f"  Middle density: {middle_density:.4f}, Edge density: {edge_density:.4f}")
    assert middle_density > edge_density, "Middle should be denser than edges"
    print("  ✓ Passed")
    
    # Test 3: Prefix LM splits
    print("\n[TEST] prefix_lm_mask")
    for mode in ["random", "short", "long"]:
        splits = [prefix_lm_mask(seq_len, mode, device)[1] for _ in range(100)]
        avg_split = sum(splits) / len(splits)
        std_split = (sum((s - avg_split)**2 for s in splits) / len(splits)) ** 0.5
        print(f"  {mode:8s}: avg_split={avg_split:.1f}, std={std_split:.1f}")
        
        if mode == "short":
            assert avg_split > 80, "Short target should have long prefix"
        elif mode == "long":
            assert avg_split < 25, "Long target should have short prefix"
    print("  ✓ Passed")
    
    # Test 4: Infilling
    print("\n[TEST] infilling_mask")
    for _ in range(50):
        mask, start, end = infilling_mask(seq_len, 0.3, device)
        assert 9 <= start, f"Hole starts too early: {start}"
        assert end <= 91, f"Hole ends too late: {end}"
        assert mask[:start].sum() == 0, "Before hole should be unmasked"
        assert mask[start:end].all(), "Hole should be fully masked"
        assert mask[end:].sum() == 0, "After hole should be unmasked"
    print(f"  Hole frac=0.3, expected size≈30")
    print("  ✓ Passed")
    
    # Test 5: Sentinel creation
    print("\n[TEST] create_sentinel_ids")
    mask = torch.tensor([False, False, True, True, False, True, True, True, False, False], device=device)
    sids = create_sentinel_ids(mask, 32099)
    
    print(f"  Mask:      {mask.int().tolist()}")
    print(f"  Sentinels: {sids.tolist()}")
    
    assert sids[2].item() == 32099, "First span should get sentinel_start"
    assert sids[3].item() == -1, "Continuation should be -1"
    assert sids[5].item() == 32098, "Second span should decrement"
    print("  ✓ Passed")
    
    # Test 6: Full collator
    print("\n[TEST] UL25DataCollator")
    
    class MockTokenizer:
        eos_token_id = 1
        pad_token_id = 0
        all_special_tokens = [f"<extra_id_{i}>" for i in range(100)]
        all_special_ids = list(range(32000, 32100))
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text[:10]]
    
    tokenizer = MockTokenizer()
    collator = UL25DataCollator(tokenizer, UL25Config.recommended(), max_length=128, max_labels_length=64)
    
    examples = [
        {"input_ids": torch.randint(100, 1000, (64,), device=device)},
        {"input_ids": torch.randint(100, 1000, (48,), device=device)},
        {"input_ids": torch.randint(100, 1000, (72,), device=device)},
    ]
    
    batch = collator(examples)
    
    assert batch["input_ids"].shape[0] == 3
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert (batch["attention_mask"].sum(dim=1) > 0).all(), "Each example should have some attention"
    
    print(f"  Batch shapes: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")
    print("  ✓ Passed")
    
    print("\n" + "=" * 60)
    print("ALL CORRECTNESS TESTS PASSED")
    print("=" * 60)


# =============================================================================
# BENCHMARKS
# =============================================================================

def benchmark_masking_functions(device: torch.device):
    """Benchmark individual masking functions."""
    print("\n" + "=" * 60)
    print(f"MASKING FUNCTION BENCHMARKS (device={device})")
    print("=" * 60)
    
    seq_lens = [128, 512, 2048, 8192]
    n_iter = 100
    
    funcs = {
        "span_corruption": lambda sl: span_corruption_mask(sl, 0.15, 3.0, 512, device),
        "middle_heavy": lambda sl: middle_heavy_mask(sl, 0.15, device),
        "prefix_lm": lambda sl: prefix_lm_mask(sl, "random", device),
        "infilling": lambda sl: infilling_mask(sl, 0.3, device),
    }
    
    results = {}
    
    print(f"\n{'Function':<20} " + " ".join(f"{sl:>10}" for sl in seq_lens))
    print("-" * (20 + 11 * len(seq_lens)))
    
    for name, func in funcs.items():
        times = []
        for sl in seq_lens:
            # Warmup
            for _ in range(5):
                _ = func(sl)
            
            with Timer(sync_cuda=(device.type == "cuda")) as t:
                for _ in range(n_iter):
                    _ = func(sl)
            
            time_ms = t.elapsed / n_iter * 1000
            times.append(time_ms)
        
        results[name] = times
        print(f"{name:<20} " + " ".join(f"{t:>9.3f}ms" for t in times))
    
    return results


def benchmark_full_collator(device: torch.device):
    """Benchmark full collator throughput."""
    print("\n" + "=" * 60)
    print(f"FULL COLLATOR BENCHMARKS (device={device})")
    print("=" * 60)
    
    class MockTokenizer:
        eos_token_id = 1
        pad_token_id = 0
        all_special_tokens = [f"<extra_id_{i}>" for i in range(100)]
        all_special_ids = list(range(32000, 32100))
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text[:10]]
    
    tokenizer = MockTokenizer()
    
    configs = {
        "recommended": UL25Config.recommended(),
        "ul2_original": UL25Config.ul2_original(),
        "t5_standard": UL25Config.t5_standard(),
    }
    
    batch_sizes = [1, 8, 32]
    seq_lens = [128, 256, 512]
    n_iter = 50
    
    print(f"\n{'Config':<15} {'Batch':<8} {'SeqLen':<8} {'Time(ms)':<12} {'Samples/s':<12} {'Tokens/s':<15}")
    print("-" * 75)
    
    for config_name, config in configs.items():
        collator = UL25DataCollator(tokenizer, config, max_length=1024, max_labels_length=256)
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                examples = [{"input_ids": torch.randint(100, 1000, (seq_len,), device=device)} 
                           for _ in range(batch_size)]
                
                # Warmup
                for _ in range(3):
                    _ = collator(examples)
                
                with Timer(sync_cuda=(device.type == "cuda")) as t:
                    for _ in range(n_iter):
                        _ = collator(examples)
                
                time_ms = t.elapsed / n_iter * 1000
                samples_per_sec = batch_size / (t.elapsed / n_iter)
                tokens_per_sec = samples_per_sec * seq_len
                
                print(f"{config_name:<15} {batch_size:<8} {seq_len:<8} {time_ms:<12.2f} {samples_per_sec:<12.0f} {tokens_per_sec:<15,.0f}")


def benchmark_cpu_vs_gpu():
    """Compare CPU vs GPU performance."""
    if not torch.cuda.is_available():
        print("\n[SKIP] CPU vs GPU benchmark - CUDA not available")
        return
    
    print("\n" + "=" * 60)
    print("CPU vs GPU COMPARISON")
    print("=" * 60)
    
    class MockTokenizer:
        eos_token_id = 1
        pad_token_id = 0
        all_special_tokens = [f"<extra_id_{i}>" for i in range(100)]
        all_special_ids = list(range(32000, 32100))
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text[:10]]
    
    tokenizer = MockTokenizer()
    config = UL25Config.recommended()
    
    batch_size = 32
    seq_len = 512
    n_iter = 50
    
    results = {}
    
    for device_name in ["cpu", "cuda"]:
        device = torch.device(device_name)
        collator = UL25DataCollator(tokenizer, config, max_length=1024, max_labels_length=256)
        examples = [{"input_ids": torch.randint(100, 1000, (seq_len,), device=device)} 
                   for _ in range(batch_size)]
        
        # Warmup
        for _ in range(5):
            _ = collator(examples)
        
        with Timer(sync_cuda=(device_name == "cuda")) as t:
            for _ in range(n_iter):
                _ = collator(examples)
        
        time_ms = t.elapsed / n_iter * 1000
        tokens_per_sec = batch_size * seq_len / (t.elapsed / n_iter)
        results[device_name] = {"time_ms": time_ms, "tokens_per_sec": tokens_per_sec}
    
    print(f"\nBatch size: {batch_size}, Seq len: {seq_len}")
    print(f"\n{'Device':<10} {'Time (ms)':<15} {'Tokens/sec':<20}")
    print("-" * 45)
    for device_name, metrics in results.items():
        print(f"{device_name:<10} {metrics['time_ms']:<15.2f} {metrics['tokens_per_sec']:<20,.0f}")
    
    if "cuda" in results and "cpu" in results:
        speedup = results["cpu"]["time_ms"] / results["cuda"]["time_ms"]
        print(f"\nGPU speedup: {speedup:.2f}x")


# =============================================================================
# REAL TOKENIZER TEST
# =============================================================================

def test_real_tokenizer():
    """Test with real HuggingFace tokenizer."""
    if not HAS_TRANSFORMERS:
        print("\n[SKIP] Real tokenizer test - transformers not installed")
        return
    
    print("\n" + "=" * 60)
    print("REAL TOKENIZER TEST")
    print("=" * 60)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")
        print(f"Loaded: google/t5-v1_1-small (vocab={tokenizer.vocab_size})")
        
        # Add UL2.5 special tokens
        tokenizer.add_special_tokens({
            "additional_special_tokens": ["[R]", "[S]", "[X]", "[I]"]
        })
        print(f"Added special tokens: [R], [S], [X], [I]")
        
        device = get_device()
        collator = UL25DataCollator(
            tokenizer=tokenizer,
            config=UL25Config.recommended(),
            max_length=128,
            max_labels_length=64,
        )
        
        # Test text
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models are transforming software development.",
            "BERT and T5 are popular transformer architectures for NLP.",
        ]
        
        examples = []
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
            examples.append({"input_ids": tokens["input_ids"].squeeze(0).to(device)})
        
        batch = collator(examples)
        
        print(f"\nBatch shapes:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        
        # Decode example
        print(f"\nExample 0:")
        print(f"  Original: {texts[0]}")
        enc_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
        print(f"  Encoder:  {enc_text[:80]}...")
        
        valid_labels = batch['labels'][0][batch['labels'][0] != -100]
        dec_text = tokenizer.decode(valid_labels, skip_special_tokens=False)
        print(f"  Decoder:  {dec_text[:80]}...")
        
        print("\n✓ Real tokenizer test passed")
        
    except Exception as e:
        print(f"\n✗ Real tokenizer test failed: {e}")


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_visualizations(device: torch.device):
    """Generate visualization plots."""
    if not HAS_MATPLOTLIB:
        print("\n[SKIP] Visualizations - matplotlib not installed")
        return
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    seq_len = 100
    n_samples = 1000
    
    # 1. Mask distribution comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor='#1a1a2e')
    
    configs = [
        ("Uniform (Standard)", lambda: span_corruption_mask(seq_len, 0.15, 3.0, 512, device)),
        ("Middle-Heavy", lambda: middle_heavy_mask(seq_len, 0.15, device)),
        ("Prefix LM (random)", lambda: prefix_lm_mask(seq_len, "random", device)[0]),
        ("Prefix LM (short)", lambda: prefix_lm_mask(seq_len, "short", device)[0]),
        ("Prefix LM (long)", lambda: prefix_lm_mask(seq_len, "long", device)[0]),
        ("Infilling", lambda: infilling_mask(seq_len, 0.3, device)[0]),
    ]
    
    for ax, (title, func) in zip(axes.flat, configs):
        ax.set_facecolor('#16213e')
        
        masks = torch.stack([func() for _ in range(n_samples)])
        avg_mask = masks.float().mean(dim=0).cpu().numpy()
        
        ax.bar(range(seq_len), avg_mask, color='#00d9ff', alpha=0.8, width=1.0)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel('Position', color='white')
        ax.set_ylabel('P(masked)', color='white')
        ax.tick_params(colors='white')
        ax.set_ylim(0, max(0.5, avg_mask.max() * 1.1))
        
        for spine in ax.spines.values():
            spine.set_color('#444')
        ax.grid(True, alpha=0.2, color='white')
    
    plt.tight_layout()
    plt.savefig('mask_distributions_torch.png', dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("  Saved: mask_distributions_torch.png")
    
    # 2. Single mask examples
    fig, axes = plt.subplots(6, 1, figsize=(14, 10), facecolor='#1a1a2e')
    
    examples = [
        ("Span (μ=3, r=0.15)", span_corruption_mask(80, 0.15, 3.0, 512, device)),
        ("Span (μ=8, r=0.25)", span_corruption_mask(80, 0.25, 8.0, 512, device)),
        ("Middle-Heavy", middle_heavy_mask(80, 0.20, device)),
        ("Prefix LM (random)", prefix_lm_mask(80, "random", device)[0]),
        ("Prefix LM (short)", prefix_lm_mask(80, "short", device)[0]),
        ("Infilling (30%)", infilling_mask(80, 0.3, device)[0]),
    ]
    
    for ax, (title, mask) in zip(axes, examples):
        ax.set_facecolor('#16213e')
        mask_np = mask.cpu().numpy()
        seq_len_ex = len(mask_np)
        
        colors = ['#00d9ff' if not m else '#ff6b6b' for m in mask_np]
        ax.bar(range(seq_len_ex), [1]*seq_len_ex, color=colors, width=1.0)
        
        n_masked = mask.sum().item()
        ax.set_ylabel(f"{title}\n({n_masked}/{seq_len_ex})", color='white', fontsize=9, 
                     rotation=0, ha='right', va='center', labelpad=60)
        ax.set_xlim(-0.5, seq_len_ex - 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.tick_params(colors='white')
        
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    axes[-1].set_xlabel('Position (blue=kept, red=masked)', color='white')
    plt.suptitle('UL2.5 Mask Examples (PyTorch)', color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mask_examples_torch.png', dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("  Saved: mask_examples_torch.png")
    
    # 3. Benchmark chart
    results = benchmark_masking_functions(device)
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    
    seq_lens = [128, 512, 2048, 8192]
    x = range(len(seq_lens))
    width = 0.2
    colors = ['#00d9ff', '#ff6b6b', '#6bcb77', '#ffd93d']
    
    for i, (name, times) in enumerate(results.items()):
        offset = (i - len(results)/2 + 0.5) * width
        ax.bar([xi + offset for xi in x], times, width, label=name, color=colors[i % len(colors)], alpha=0.8)
    
    ax.set_xlabel('Sequence Length', color='white')
    ax.set_ylabel('Time (ms)', color='white')
    ax.set_title(f'Masking Function Benchmark ({device})', color='white', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lens)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#16213e', edgecolor='white', labelcolor='white')
    ax.set_yscale('log')
    
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.grid(True, alpha=0.2, color='white', axis='y')
    
    plt.tight_layout()
    plt.savefig('benchmark_torch.png', dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("  Saved: benchmark_torch.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("UL2.5 PyTorch/HF Collator - Benchmark Suite")
    print("=" * 60)
    
    device = get_device()
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    
    # Run tests
    test_masking_correctness(device)
    benchmark_masking_functions(device)
    benchmark_full_collator(device)
    benchmark_cpu_vs_gpu()
    test_real_tokenizer()
    create_visualizations(device)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    
    # Summary
    print("""
SUMMARY
-------
Files generated (if matplotlib installed):
  - mask_distributions_torch.png
  - mask_examples_torch.png  
  - benchmark_torch.png

Key metrics to compare with NumPy baseline:
  - Masking function times at seq_len=512
  - Full collator throughput (tokens/sec)
  - GPU speedup (if CUDA available)
""")


if __name__ == "__main__":
    main()
