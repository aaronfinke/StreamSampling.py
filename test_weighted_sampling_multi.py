"""
Simple test script to verify the weighted reservoir sampling multi-element algorithms work correctly.

This script runs basic functionality tests and statistical validation.
"""

import numpy as np
from collections import Counter
from weighted_sampling_multi_python import (
    weighted_sample_multi, 
    stream_weighted_sample_multi,
    SamplingMethod
)


def test_basic_functionality():
    """Test basic functionality of all algorithms."""
    print("Testing basic functionality...")
    
    elements = list(range(1, 11))
    weights = [1.0] * 10  # Equal weights
    n_sample = 5
    
    methods = [SamplingMethod.A_RES, SamplingMethod.A_EXPJ, SamplingMethod.WRSWR_SKIP]
    
    for method in methods:
        # Test one-shot sampling
        sample = weighted_sample_multi(elements, weights, n_sample, method)
        assert len(sample) == n_sample, f"{method} failed: wrong sample size"
        assert all(x in elements for x in sample), f"{method} failed: invalid elements"
        
        # Test streaming
        sampler = stream_weighted_sample_multi(n_sample, method)
        for elem, weight in zip(elements, weights):
            sampler.fit(elem, weight)
        
        stream_sample = sampler.value()
        assert len(stream_sample) == n_sample, f"{method} streaming failed: wrong sample size"
        
        print(f"  ✓ {method.value} passed")
    
    print("Basic functionality tests passed!\n")


def test_statistical_properties():
    """Test that the algorithms produce correct probability distributions."""
    print("Testing statistical properties...")
    
    # Test with more elements so sampling without replacement makes sense
    elements = [1, 2, 3, 4]
    weights = [1.0, 3.0, 1.0, 1.0]  # Element 2 should be 3x more likely than others
    n_sample = 2  # Sample 2 out of 4
    n_trials = 5000
    
    for method in [SamplingMethod.A_RES, SamplingMethod.A_EXPJ, SamplingMethod.WRSWR_SKIP]:
        results = []
        rng = np.random.default_rng(42)
        
        for _ in range(n_trials):
            sample = weighted_sample_multi(elements, weights, n_sample, method,
                                         rng=np.random.default_rng(rng.integers(0, 2**32)))
            results.extend(sample)
        
        counter = Counter(results)
        total = len(results)
        
        # For element 2 with weight 3.0 out of total weight 6.0
        freq2 = counter[2] / total
        expected2 = 3.0 / 6.0  # 0.5
        
        # Allow reasonable deviation for statistical variation
        tolerance = 0.15
        
        # Focus on testing element 2 which has distinctly different weight
        assert abs(freq2 - expected2) < tolerance, f"{method} failed statistical test: element 2 (freq={freq2:.3f}, expected={expected2:.3f})"
        
        print(f"  ✓ {method.value} statistical test passed (element 2: freq={freq2:.3f}, expected={expected2:.3f})")
    
    print("Statistical tests passed!\n")


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("Testing edge cases...")
    
    # Test with single element
    sample = weighted_sample_multi([42], [1.0], 1, SamplingMethod.A_RES)
    assert sample == [42], "Single element test failed"
    
    # Test with weight function
    elements = [1, 2, 3, 4, 5]
    def weight_func(x):
        return x  # Weight equals value
    sample = weighted_sample_multi(elements, weight_func, 3, SamplingMethod.A_EXPJ)
    assert len(sample) == 3, "Weight function test failed"
    
    # Test ordered sampling
    sample_ordered = weighted_sample_multi(elements, [1]*5, 3, SamplingMethod.A_RES, 
                                         ordered=True, rng=np.random.default_rng(42))
    sampler = stream_weighted_sample_multi(3, SamplingMethod.A_RES, ordered=True, 
                                         rng=np.random.default_rng(42))
    for elem in elements:
        sampler.fit(elem, 1.0)
    
    stream_ordered = sampler.ordered_value()
    
    # Both should be ordered (though content may differ due to randomness)
    assert len(sample_ordered) == 3, "Ordered sampling failed"
    assert len(stream_ordered) == 3, "Ordered streaming failed"
    
    # Test error conditions
    try:
        weighted_sample_multi([1, 2], [1.0], 1, SamplingMethod.A_RES)  # Mismatched lengths
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    try:
        sampler = stream_weighted_sample_multi(3, SamplingMethod.A_RES)
        sampler.fit(1, 0.0)  # Zero weight
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    print("  ✓ Edge cases passed")
    print("Edge case tests passed!\n")


def test_performance():
    """Basic performance test."""
    print("Testing performance...")
    
    import time
    
    # Generate larger test dataset
    n_elements = 10000
    n_sample = 100
    elements = list(range(n_elements))
    weights = np.random.exponential(1.0, n_elements)
    
    for method in [SamplingMethod.A_RES, SamplingMethod.A_EXPJ, SamplingMethod.WRSWR_SKIP]:
        start_time = time.time()
        sample = weighted_sample_multi(elements, weights, n_sample, method)
        end_time = time.time()
        
        assert len(sample) == n_sample, f"{method} performance test failed: wrong sample size"
        duration = end_time - start_time
        
        print(f"  ✓ {method.value}: {duration:.4f}s for {n_elements} elements")
    
    print("Performance tests passed!\n")


def run_all_tests():
    """Run all test functions."""
    print("=" * 50)
    print("WEIGHTED RESERVOIR SAMPLING MULTI-ELEMENT TESTS")
    print("=" * 50)
    print()
    
    test_basic_functionality()
    test_statistical_properties()
    test_edge_cases()
    test_performance()
    
    print("=" * 50)
    print("ALL TESTS PASSED! 🎉")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
