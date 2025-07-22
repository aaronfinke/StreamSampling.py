"""Quick debug script to see what's happening with A-Res algorithm."""

import numpy as np
from collections import Counter
from weighted_sampling_multi_python import weighted_sample_multi, SamplingMethod

# Simple test case
elements = [1, 2]
weights = [1.0, 3.0]  # Element 2 should be 3x more likely
n_sample = 2
n_trials = 100

print("Debug: A-Res algorithm with elements [1, 2] and weights [1.0, 3.0]")
print("Expected frequencies: element 1 = 0.25, element 2 = 0.75")
print()

results = []
rng = np.random.default_rng(42)

for i in range(n_trials):
    sample = weighted_sample_multi(elements, weights, n_sample, SamplingMethod.A_RES,
                                 rng=np.random.default_rng(rng.integers(0, 2**32)))
    results.extend(sample)
    if i < 10:  # Show first 10 samples
        print(f"Sample {i+1}: {sample}")

counter = Counter(results)
total = len(results)
freq1 = counter[1] / total
freq2 = counter[2] / total

print(f"\nResults after {n_trials} trials:")
print(f"Element 1: {counter[1]} occurrences, frequency = {freq1:.3f}")
print(f"Element 2: {counter[2]} occurrences, frequency = {freq2:.3f}")
print(f"Total samples: {total}")

# Test with WRSWR-SKIP for comparison
print("\nComparing with WRSWR-SKIP:")
results_wrswr = []
rng = np.random.default_rng(42)

for i in range(n_trials):
    sample = weighted_sample_multi(elements, weights, n_sample, SamplingMethod.WRSWR_SKIP,
                                 rng=np.random.default_rng(rng.integers(0, 2**32)))
    results_wrswr.extend(sample)

counter_wrswr = Counter(results_wrswr)
freq1_wrswr = counter_wrswr[1] / len(results_wrswr)
freq2_wrswr = counter_wrswr[2] / len(results_wrswr)

print(f"WRSWR - Element 1: frequency = {freq1_wrswr:.3f}")
print(f"WRSWR - Element 2: frequency = {freq2_wrswr:.3f}")
