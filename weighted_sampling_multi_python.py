"""
Python/NumPy implementation of Weighted Reservoir Sampling algorithm for multiple elements.

This is a conversion of the Julia algorithms from StreamSampling.jl implementing
weighted reservoir sampling algorithm that samples multiple elements:

WRSWR-SKIP - Weighted Reservoir Sampling with Replacement SKIP for multiple elements

Original Julia implementation: https://github.com/JuliaDynamics/StreamSampling.jl
"""

import numpy as np
from typing import Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
from scipy.stats import binom


class SamplingMethod(Enum):
    """Available weighted sampling methods."""
    WRSWR_SKIP = "wrswr_skip"


@dataclass
class WeightedReservoirSamplerMulti:
    """
    Weighted Reservoir Sampling for multiple elements using WRSWR-SKIP algorithm.
    
    This class implements the WRSWR-SKIP algorithm for weighted reservoir sampling
    that maintains a fixed-size sample from a weighted stream.
    
    Attributes:
        n (int): Number of elements to sample
        method (SamplingMethod): Which algorithm to use (WRSWR_SKIP)
        ordered (bool): Whether to maintain insertion order
        seen_k (int): Number of elements seen so far
        rng (np.random.Generator): Random number generator
        
    WRSWR-SKIP attributes:
        state (float): Cumulative weight
        skip_w (float): Skip threshold
        weights (np.ndarray): Weight storage during reservoir filling
        values (np.ndarray): Element storage
        order_indices (np.ndarray): Order tracking for ordered sampling
    """
    
    def __init__(self, n: int, method: SamplingMethod = SamplingMethod.WRSWR_SKIP, 
                 ordered: bool = False, rng: Optional[np.random.Generator] = None):
        """
        Initialize the weighted reservoir sampler.
        
        Args:
            n: Number of elements to sample
            method: Which sampling algorithm to use
            ordered: Whether to maintain insertion order of elements
            rng: Random number generator. If None, uses default numpy generator.
        """
        if n <= 0:
            raise ValueError("Sample size n must be positive")
            
        self.n = n
        self.method = method
        self.ordered = ordered
        self.seen_k = 0
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # WRSWR-SKIP initialization
        # Vector-based storage for WRSWR-SKIP
        self.state = 0.0  # Cumulative weight
        self.skip_w = 0.0  # Skip threshold
        self.weights = np.zeros(n)
        self.values = np.empty(n, dtype=object)
        self.order_indices = np.arange(n) if ordered else None
    
    def fit(self, element: Any, weight: float) -> None:
        """
        Update the reservoir sample with a new weighted element.
        
        Args:
            element: The new element to consider for sampling
            weight: The weight of the element (must be positive)
        
        Raises:
            ValueError: If weight is not positive
        """
        if weight <= 0:
            raise ValueError("Weight must be positive")
        
        self.seen_k += 1
        self._fit_wrswr_skip(element, weight)
    
    def _fit_wrswr_skip(self, element: Any, weight: float) -> None:
        """Implement WRSWR-SKIP algorithm."""
        self.state += weight  # Cumulative weight
        
        if self.seen_k <= self.n:
            # Still filling the reservoir
            idx = self.seen_k - 1  # Convert to 0-based indexing
            self.values[idx] = element
            self.weights[idx] = weight
            
            if self.seen_k == self.n:
                # Reservoir is now full, resample and compute skip
                indices = self.rng.choice(self.n, size=self.n, replace=True, 
                                        p=self.weights / self.state)
                self.values[:] = self.values[indices]
                if self.ordered:
                    # Update order indices
                    self.order_indices = np.arange(self.n)
                self._recompute_skip_wrswr(self.n)
                # Clear weights array as it's no longer needed
                self.weights = np.zeros(self.n)
        else:
            # Reservoir is full, check skip condition
            if self.skip_w <= self.state:
                # Time to update
                p = weight / self.state
                z = np.exp((self.n - 4) * np.log1p(-p))
                lower_bound = z * (1 - p) ** 4
                c = self.rng.uniform(lower_bound, 1.0)
                k = self._choose(self.n, p, c, z)
                
                # Replace k random elements
                for j in range(k):
                    r = self.rng.integers(j, self.n)
                    self.values[r], self.values[j] = element, self.values[r]
                    if self.ordered and self.order_indices is not None:
                        self.order_indices[r], self.order_indices[j] = \
                            self.seen_k, self.order_indices[r]
                
                self._recompute_skip_wrswr(self.n)
    
    def _recompute_skip_wrswr(self, n: int) -> None:
        """Recompute skip threshold for WRSWR-SKIP."""
        q = np.exp(-self.rng.exponential() / n)
        self.skip_w = self.state / q
    
    def _choose(self, n: int, p: float, c: float, z: float) -> int:
        """
        Choose number of elements to replace in WRSWR-SKIP.
        
        This approximates the binomial distribution for small values
        and falls back to the exact quantile for larger values.
        """
        q = 1 - p
        
        # Try small values first (fast approximation)
        k = z * q**3 * (q + n * p)
        if k > c:
            return 1
            
        k += n * p * (n - 1) * p * z * q**2 / 2
        if k > c:
            return 2
            
        k += n * p * (n - 1) * p * (n - 2) * p * z * q / 6
        if k > c:
            return 3
            
        k += n * p * (n - 1) * p * (n - 2) * p * (n - 3) * p * z / 24
        if k > c:
            return 4
        
        # Fall back to exact binomial quantile
        result = binom.ppf(c, n, p)
        return int(result)
    
    def value(self) -> List[Any]:
        """
        Get the current sampled values.
        
        Returns:
            List of currently sampled elements. For WRSWR-SKIP with fewer than n
            elements seen, may resample to reach the target size.
        """
        if self.seen_k == 0:
            return []
        
        if self.seen_k < self.n:
            if self.seen_k == 0:
                return []
            # Resample from what we have so far to reach target size
            valid_weights = self.weights[:self.seen_k]
            if np.sum(valid_weights) > 0:
                indices = self.rng.choice(self.seen_k, size=self.n, replace=True,
                                        p=valid_weights / np.sum(valid_weights))
                return [self.values[i] for i in indices]
            else:
                return list(self.values[:self.seen_k])
        else:
            return list(self.values)
    
    def ordered_value(self) -> List[Any]:
        """
        Get the current sampled values in insertion order.
        
        Returns:
            List of currently sampled elements in the order they were inserted.
            Only meaningful if ordered=True was set during initialization.
        """
        if not self.ordered:
            raise ValueError("Ordered sampling was not enabled. Set ordered=True during initialization.")
        
        if self.seen_k == 0:
            return []
        
        if self.seen_k < self.n:
            # For partial fills, just return in current order
            return list(self.values[:self.seen_k])
        else:
            # Sort by order indices
            if self.order_indices is not None:
                sorted_indices = np.argsort(self.order_indices)
                return [self.values[i] for i in sorted_indices]
            else:
                return list(self.values)
    
    def empty(self) -> None:
        """Reset the sampler to its initial state."""
        self.seen_k = 0
        self.state = 0.0
        self.skip_w = 0.0
        self.weights = np.zeros(self.n)
        self.values = np.empty(self.n, dtype=object)
        if self.ordered:
            self.order_indices = np.arange(self.n)
    
    def nobs(self) -> int:
        """Get the number of elements observed so far."""
        return self.seen_k


def weighted_sample_multi(
    elements: Union[List, np.ndarray], 
    weights: Union[List, np.ndarray, Callable], 
    n: int,
    method: SamplingMethod = SamplingMethod.WRSWR_SKIP,
    ordered: bool = False,
    rng: Optional[np.random.Generator] = None
) -> List[Any]:
    """
    Sample multiple elements from a weighted collection using reservoir sampling.
    
    Args:
        elements: Collection of elements to sample from
        weights: Either a collection of weights (same length as elements) or 
                a callable that takes an element and returns its weight
        n: Number of elements to sample
        method: Which sampling algorithm to use
        ordered: Whether to return elements in insertion order
        rng: Random number generator. If None, uses default numpy generator.
    
    Returns:
        List of n sampled elements from the collection
    
    Example:
        >>> elements = ['a', 'b', 'c', 'd', 'e']
        >>> weights = [0.1, 0.2, 0.3, 0.2, 0.2]
        >>> sample = weighted_sample_multi(elements, weights, n=3)
        >>> len(sample)
        3
        
        >>> # Using WRSWR-SKIP algorithm with weight function
        >>> weight_func = lambda x: ord(x) - ord('a') + 1  # a=1, b=2, etc.
        >>> sample = weighted_sample_multi(elements, weight_func, n=2, 
        ...                               method=SamplingMethod.WRSWR_SKIP)
    """
    sampler = WeightedReservoirSamplerMulti(n, method, ordered, rng)
    
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
    
    return sampler.ordered_value() if ordered else sampler.value()


def stream_weighted_sample_multi(
    n: int,
    method: SamplingMethod = SamplingMethod.WRSWR_SKIP,
    ordered: bool = False,
    rng: Optional[np.random.Generator] = None
) -> WeightedReservoirSamplerMulti:
    """
    Create a streaming weighted reservoir sampler for multiple elements.
    
    Args:
        n: Number of elements to sample
        method: Which sampling algorithm to use
        ordered: Whether to maintain insertion order
        rng: Random number generator. If None, uses default numpy generator.
    
    Returns:
        A WeightedReservoirSamplerMulti instance ready for streaming updates
    
    Example:
        >>> sampler = stream_weighted_sample_multi(n=3, method=SamplingMethod.WRSWR_SKIP)
        >>> 
        >>> # Process elements as they arrive in a stream
        >>> for element, weight in stream_of_weighted_elements():
        ...     sampler.fit(element, weight)
        ...     current_sample = sampler.value()
        ...     print(f"Current sample: {current_sample}")
    """
    return WeightedReservoirSamplerMulti(n, method, ordered, rng)


# Example usage and test functions
if __name__ == "__main__":
    from collections import Counter
    import time
    
    def test_algorithms():
        """Test WRSWR-SKIP algorithm with known weights."""
        print("=== Testing Weighted Reservoir Sampling Algorithm ===\n")
        
        # Test data
        elements = list(range(1, 11))  # 1 to 10
        weights = [1.0 if x <= 5 else 2.0 for x in elements]  # 1-5: weight=1, 6-10: weight=2
        n_sample = 3
        n_trials = 5000
        
        methods = [
            (SamplingMethod.WRSWR_SKIP, "WRSWR-SKIP")
        ]
        
        for method, method_name in methods:
            print(f"Testing {method_name}:")
            results = []
            rng = np.random.default_rng(42)
            
            for _ in range(n_trials):
                sample = weighted_sample_multi(elements, weights, n_sample, method,
                                             rng=np.random.default_rng(rng.integers(0, 2**32)))
                results.extend(sample)
            
            # Count frequencies
            counter = Counter(results)
            total = sum(counter.values())
            
            print(f"  Results from {n_trials} trials (sample size {n_sample}):")
            print("  Element | Count | Frequency | Weight")
            print("  " + "-" * 35)
            
            for element in elements:
                count = counter.get(element, 0)
                frequency = count / total if total > 0 else 0
                weight = weights[element - 1]
                print(f"  {element:7} | {count:5} | {frequency:9.3f} | {weight:6.1f}")
            print()
    
    def test_streaming():
        """Test the streaming interface."""
        print("=== Testing Streaming Interface ===\n")
        
        sampler = stream_weighted_sample_multi(n=3, method=SamplingMethod.WRSWR_SKIP)
        
        # Simulate a stream of weighted elements
        stream_elements = [(f"item_{i}", np.random.exponential(1.0)) for i in range(10)]
        
        print("Processing stream with WRSWR-SKIP:")
        for element, weight in stream_elements:
            sampler.fit(element, weight)
            current_sample = sampler.value()
            print(f"  Added {element} (weight={weight:.3f}), current sample: {current_sample}")
        
        print(f"Final sample: {sampler.value()}")
        print(f"Total elements processed: {sampler.nobs()}")
        print()
    
    def test_ordered_sampling():
        """Test ordered sampling."""
        print("=== Testing Ordered Sampling ===\n")
        
        elements = ['first', 'second', 'third', 'fourth', 'fifth']
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # Equal weights
        
        # Test with ordered=True
        sample_ordered = weighted_sample_multi(elements, weights, n=3, ordered=True,
                                             rng=np.random.default_rng(42))
        
        # Test with ordered=False  
        sample_unordered = weighted_sample_multi(elements, weights, n=3, ordered=False,
                                               rng=np.random.default_rng(42))
        
        print(f"Original elements: {elements}")
        print(f"Ordered sample: {sample_ordered}")
        print(f"Unordered sample: {sample_unordered}")
        print()
    
    def test_performance():
        """Test performance with larger datasets."""
        print("=== Performance Testing ===\n")
        
        # Generate large dataset
        n_items = 100000
        rng = np.random.default_rng(42)
        elements = np.arange(n_items)
        weights = rng.exponential(1.0, n_items)
        n_sample = 1000
        
        methods = [
            (SamplingMethod.WRSWR_SKIP, "WRSWR-SKIP")
        ]
        
        print(f"Sampling {n_sample} elements from {n_items} items:")
        for method, method_name in methods:
            start_time = time.time()
            sample = weighted_sample_multi(elements, weights, n_sample, method, rng=rng)
            end_time = time.time()
            
            print(f"  {method_name}: {end_time - start_time:.4f} seconds")
            print(f"    Sample size: {len(sample)}")
            print(f"    Min element: {min(sample)}")
            print(f"    Max element: {max(sample)}")
        print()
    
    # Run all tests
    test_algorithms()
    test_streaming()
    test_ordered_sampling()
    test_performance()
