# Weighted Reservoir Sampling with Replacement (Python/NumPy Implementation)

This repository contains a Python/NumPy implementation of the **Weighted Reservoir Sampling with Replacement SKIP** algorithm, converted from the Julia implementation in [StreamSampling.jl](https://github.com/JuliaDynamics/StreamSampling.jl).

## Algorithm Overview

The WRSWR-SKIP algorithm implements weighted random reservoir sampling with replacement for single element sampling from a stream. This is particularly useful when:

- You need to sample from a stream of weighted data where elements arrive one at a time
- You don't know the total number of elements in advance
- Each element has an associated weight that affects its probability of being selected
- You want constant memory usage regardless of stream size

The algorithm is adapted from ["Weighted Reservoir Sampling with Replacement from Multiple Data Streams, A. Meligrana, 2024"](https://arxiv.org/html/2403.20256v3).

### Key Features

- **Streaming**: Processes elements one at a time without storing the entire dataset
- **Weighted**: Each element can have a different probability weight
- **Constant Memory**: O(1) memory usage regardless of stream size
- **Single Pass**: Only requires one pass through the data
- **Mathematically Sound**: Provides correct(ish) probability distributions

## Algorithm Details

The core algorithm maintains:
- `seen_k`: Number of elements processed
- `total_w`: Cumulative sum of all weights
- `skip_w`: Weight threshold for next update
- `current_value`: Currently stored sample

For each new element `(element, weight)`:
1. Increment `seen_k` and add `weight` to `total_w`
2. If `skip_w <= total_w`:
   - Update `skip_w = total_w / random()`
   - Store `element` as the new sample

This creates the correct weighted probability distribution where element selection probability is proportional to its weight.

## Installation and Usage

### Requirements

```python
import numpy as np
from typing import Any, Optional, Union
from dataclasses import dataclass
from collections import Counter  # For examples only
```

### Basic Usage

```python
from weighted_sampling_single_python import weighted_sample_single, stream_weighted_sample_single

# Example 1: One-shot sampling
elements = ['apple', 'banana', 'cherry', 'date']
weights = [0.1, 0.3, 0.4, 0.2]  # Cherry is most likely
sample = weighted_sample_single(elements, weights)
print(f"Selected: {sample}")

# Example 2: Using a weight function
numbers = range(1, 11)
def weight_func(x):
    return 1.0 / x  # Smaller numbers get higher weights

sample = weighted_sample_single(numbers, weight_func)
print(f"Selected: {sample}")

# Example 3: Streaming/online processing
sampler = stream_weighted_sample_single()
for element, weight in data_stream:
    sampler.fit(element, weight)
    current_sample = sampler.value()
    print(f"Current sample: {current_sample}")
```

### Advanced Examples

See `examples_weighted_sampling.py` for comprehensive examples including:
- NumPy array handling
- Large-scale data processing
- Comparison with NumPy's `choice()` function
- Practical applications (log sampling, feature selection, customer sampling)

## API Reference

### Classes

#### `WeightedReservoirSamplerSingle`

The main sampler class for streaming scenarios.

**Constructor:**
```python
WeightedReservoirSamplerSingle(rng: Optional[np.random.Generator] = None)
```

**Methods:**
- `fit(element, weight)`: Add a weighted element to the stream
- `value()`: Get the current sample (or None if no elements processed)
- `empty()`: Reset the sampler to initial state
- `nobs()`: Get the number of elements processed

### Functions

#### `weighted_sample_single()`

One-shot sampling from a collection of weighted elements.

```python
weighted_sample_single(
    elements: Union[list, np.ndarray], 
    weights: Union[list, np.ndarray, callable], 
    rng: Optional[np.random.Generator] = None
) -> Any
```

**Parameters:**
- `elements`: Collection of elements to sample from
- `weights`: Either weights array (same length as elements) or weight function
- `rng`: Random number generator (optional)

#### `stream_weighted_sample_single()`

Create a streaming sampler instance.

```python
stream_weighted_sample_single(
    rng: Optional[np.random.Generator] = None
) -> WeightedReservoirSamplerSingle
```

## Comparison with Original Julia Implementation

| Aspect | Julia (Original) | Python (This Implementation) |
|--------|------------------|-------------------------------|
| **Core Algorithm** | ✓ Identical | ✓ Identical |
| **Memory Usage** | O(1) | O(1) |
| **Time Complexity** | O(1) per element | O(1) per element |
| **Type System** | Generic types | Any type with type hints |
| **RNG** | AbstractRNG | numpy.random.Generator |
| **Mutability** | Hybrid struct variants | Single class with mutable state |
| **Interface** | OnlineStatsBase | Simple fit/value methods |

### Key Differences

1. **Type System**: Julia's generic type system vs Python's dynamic typing with hints
2. **Memory Management**: Julia's hybrid structs vs Python's standard classes
3. **RNG**: Julia's AbstractRNG vs NumPy's Generator
4. **Interface**: Julia's OnlineStatsBase protocol vs simple method names


## Testing

Run the test suite:

```bash
python weighted_sampling_single_python.py
```

Run comprehensive examples:

```bash
python examples_weighted_sampling.py
```

The tests verify:
- Correct probability distributions
- Streaming interface functionality
- Edge cases and error handling
- Performance characteristics

## References

1. **"Weighted Reservoir Sampling with Replacement from Multiple Data Streams"** - A. Meligrana, 2024
2. **StreamSampling.jl** - Original Julia implementation: https://github.com/JuliaDynamics/StreamSampling.jl
3. **"Random Sampling with a Reservoir"** - J. S. Vitter, 1985 (foundational reservoir sampling)

## License

This implementation follows the same principles as the original StreamSampling.jl package. Please refer to the original repository for licensing information.

## Contributing

Feel free to submit issues, feature requests, or pull requests. When contributing:

1. Maintain algorithmic correctness with the original Julia implementation
2. Add appropriate tests for new features
3. Follow Python coding standards (PEP 8)
4. Update documentation as needed
