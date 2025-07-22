"""
Python/NumPy implementation of the Weighted Reservoir Sampling with Replacement SKIP algorithm.

This is a conversion of the Julia algorithm AlgWRSWRSKIP from StreamSampling.jl.
The algorithm implements weighted random reservoir sampling with replacement for a single element.

Adapted from algorithm WRSWR-SKIP described in "Weighted Reservoir Sampling with Replacement 
from Multiple Data Streams, A. Meligrana, 2024".

Original Julia implementation: https://github.com/JuliaDynamics/StreamSampling.jl
"""

import numpy as np
from typing import Any, Optional, Union
from dataclasses import dataclass


@dataclass
class WeightedReservoirSamplerSingle:
    """
    Weighted Reservoir Sampling with Replacement SKIP algorithm for single element sampling.
    
    This class implements a streaming algorithm that maintains a single sampled element
    from a weighted stream, where each element has an associated weight that affects
    its probability of being selected.
    
    Attributes:
        seen_k (int): Number of elements seen so far
        total_w (float): Cumulative sum of all weights seen
        skip_w (float): Weight threshold for next update
        rng (np.random.Generator): Random number generator
        current_value (Any): Currently stored sample value
    """
    
    def __init__(self, rng: Optional[np.random.Generator] = None):
        """
        Initialize the weighted reservoir sampler.
        
        Args:
            rng: Random number generator. If None, uses default numpy generator.
        """
        self.seen_k: int = 0
        self.total_w: float = 0.0
        self.skip_w: float = 0.0
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng()
        self.current_value: Any = None
    
    def fit(self, element: Any, weight: float) -> None:
        """
        Update the reservoir sample with a new weighted element.
        
        This is the core algorithm that decides whether to replace the current
        sample with the new element based on its weight.
        
        Args:
            element: The new element to consider for sampling
            weight: The weight of the element (must be positive)
        
        Raises:
            ValueError: If weight is not positive
        """
        if weight <= 0:
            raise ValueError("Weight must be positive")
        
        self.seen_k += 1
        self.total_w += weight
        
        # Check if we should update the sample
        if self.skip_w <= self.total_w:
            # Update skip threshold using exponential distribution
            self.skip_w = self.total_w / self.rng.random()
            # Store the new element
            self.current_value = element
    
    def value(self) -> Optional[Any]:
        """
        Get the current sampled value.
        
        Returns:
            The currently sampled element, or None if no elements have been processed.
        """
        if self.seen_k == 0:
            return None
        return self.current_value
    
    def empty(self) -> None:
        """Reset the sampler to its initial state."""
        self.seen_k = 0
        self.total_w = 0.0
        self.skip_w = 0.0
        self.current_value = None
    
    def nobs(self) -> int:
        """Get the number of elements observed so far."""
        return self.seen_k


def weighted_sample_single(
    elements: Union[list, np.ndarray], 
    weights: Union[list, np.ndarray, callable], 
    rng: Optional[np.random.Generator] = None
) -> Any:
    """
    Sample a single element from a weighted collection using reservoir sampling.
    
    This function provides a convenient interface for one-shot sampling from
    a collection of weighted elements.
    
    Args:
        elements: Collection of elements to sample from
        weights: Either a collection of weights (same length as elements) or 
                a callable that takes an element and returns its weight
        rng: Random number generator. If None, uses default numpy generator.
    
    Returns:
        A single sampled element from the collection
    
    Example:
        >>> elements = [1, 2, 3, 4, 5]
        >>> weights = [0.1, 0.2, 0.3, 0.2, 0.2]
        >>> sample = weighted_sample_single(elements, weights)
        >>> print(sample)  # One of 1, 2, 3, 4, 5 with probability proportional to weights
        
        >>> # Using a weight function
        >>> weight_func = lambda x: x  # Elements weighted by their value
        >>> sample = weighted_sample_single(elements, weight_func)
    """
    sampler = WeightedReservoirSamplerSingle(rng)
    
    # Handle weight function vs weight array
    if callable(weights):
        weight_func = weights
        for element in elements:
            weight = weight_func(element)
            sampler.fit(element, weight)
    else:
        weights = np.asarray(weights)
        if len(elements) != len(weights):
            raise ValueError("Elements and weights must have the same length")
        
        for element, weight in zip(elements, weights):
            sampler.fit(element, weight)
    
    return sampler.value()


def stream_weighted_sample_single(
    rng: Optional[np.random.Generator] = None
) -> WeightedReservoirSamplerSingle:
    """
    Create a streaming weighted reservoir sampler for single element sampling.
    
    This function returns a sampler object that can be used for online/streaming
    scenarios where elements arrive one at a time.
    
    Args:
        rng: Random number generator. If None, uses default numpy generator.
    
    Returns:
        A WeightedReservoirSamplerSingle instance ready for streaming updates
    
    Example:
        >>> sampler = stream_weighted_sample_single()
        >>> 
        >>> # Process elements as they arrive in a stream
        >>> for element, weight in stream_of_weighted_elements():
        ...     sampler.fit(element, weight)
        ...     current_sample = sampler.value()
        ...     print(f"Current sample: {current_sample}")
    """
    return WeightedReservoirSamplerSingle(rng)


# Example usage and test functions
if __name__ == "__main__":
    from collections import Counter
    
    def test_weighted_sampling():
        """Test the weighted sampling algorithm with known weights."""
        print("Testing Weighted Reservoir Sampling...")
        
        # Test elements with different weights
        elements = [1, 2, 3, 4, 5]
        weights = [0.1, 0.2, 0.3, 0.2, 0.2]  # Element 3 has highest weight
        
        # Run multiple samples to check distribution
        num_trials = 10000
        results = []
        
        rng = np.random.default_rng(42)  # For reproducible results
        
        for _ in range(num_trials):
            sample = weighted_sample_single(elements, weights, 
                                          np.random.default_rng(rng.integers(0, 2**32)))
            results.append(sample)
        
        # Count frequencies
        counter = Counter(results)
        total = sum(counter.values())
        
        print("Results from", num_trials, "trials:")
        print("Element | Count | Frequency | Expected")
        print("-" * 40)
        
        for i, (element, weight) in enumerate(zip(elements, weights)):
            count = counter[element]
            frequency = count / total
            print(f"{element:7} | {count:5} | {frequency:9.3f} | {weight:8.3f}")
        
        # Test with weight function
        print("\nTesting with weight function (element value as weight):")
        def weight_func(x):
            return x  # Weight proportional to element value
        total_weight = sum(elements)
        expected_probs = [x / total_weight for x in elements]
        
        results2 = []
        for _ in range(num_trials):
            sample = weighted_sample_single(elements, weight_func,
                                          np.random.default_rng(rng.integers(0, 2**32)))
            results2.append(sample)
        
        counter2 = Counter(results2)
        
        print("Element | Count | Frequency | Expected")
        print("-" * 40)
        
        for element, expected_prob in zip(elements, expected_probs):
            count = counter2[element]
            frequency = count / total
            print(f"{element:7} | {count:5} | {frequency:9.3f} | {expected_prob:8.3f}")
    
    def test_streaming():
        """Test the streaming interface."""
        print("\nTesting streaming interface...")
        
        sampler = stream_weighted_sample_single(np.random.default_rng(42))
        
        # Simulate a stream of weighted elements
        stream_elements = [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.2), (5, 0.2)]
        
        print("Processing stream:")
        for element, weight in stream_elements:
            sampler.fit(element, weight)
            print(f"Added element {element} with weight {weight}, current sample: {sampler.value()}")
        
        print(f"Final sample: {sampler.value()}")
        print(f"Total elements processed: {sampler.nobs()}")
    
    # Run tests
    test_weighted_sampling()
    test_streaming()
