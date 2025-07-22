"""
Extended examples and usage patterns for the Weighted Reservoir Sampling implementation.

This file demonstrates various use cases and provides additional utilities
for working with the weighted reservoir sampling algorithm.
"""

import numpy as np
from weighted_sampling_single_python import weighted_sample_single, stream_weighted_sample_single


def example_basic_usage():
    """Basic usage examples."""
    print("=== Basic Usage Examples ===\n")
    
    # Example 1: Simple list with weights
    print("1. Simple list with explicit weights:")
    items = ['apple', 'banana', 'cherry', 'date', 'elderberry']
    weights = [0.1, 0.3, 0.4, 0.1, 0.1]  # Cherry is most likely
    
    sample = weighted_sample_single(items, weights)
    print(f"   Items: {items}")
    print(f"   Weights: {weights}")
    print(f"   Sample: {sample}")
    print()
    
    # Example 2: NumPy arrays
    print("2. NumPy arrays:")
    data = np.array([10, 20, 30, 40, 50])
    weights = np.array([0.5, 0.2, 0.1, 0.1, 0.1])  # 10 is most likely
    
    sample = weighted_sample_single(data, weights)
    print(f"   Data: {data}")
    print(f"   Weights: {weights}")
    print(f"   Sample: {sample}")
    print()
    
    # Example 3: Weight function
    print("3. Using a weight function:")
    numbers = range(1, 11)  # 1 to 10
    
    def inverse_weight(x):
        """Smaller numbers get higher weights."""
        return 1.0 / x
    
    sample = weighted_sample_single(numbers, inverse_weight)
    print(f"   Numbers: {list(numbers)}")
    print("   Weight function: 1/x (smaller numbers more likely)")
    print(f"   Sample: {sample}")
    print()


def example_streaming():
    """Streaming/online usage examples."""
    print("=== Streaming Examples ===\n")
    
    # Example 1: Processing data as it arrives
    print("1. Online data processing:")
    sampler = stream_weighted_sample_single(np.random.default_rng(42))
    
    # Simulate incoming data stream
    data_stream = [
        (100, 0.1), (200, 0.2), (300, 0.3), (400, 0.2), (500, 0.2),
        (600, 0.1), (700, 0.4), (800, 0.1), (900, 0.1), (1000, 0.1)
    ]
    
    print("   Processing stream:")
    for value, weight in data_stream:
        sampler.fit(value, weight)
        print(f"   Added {value} (weight={weight}), current sample: {sampler.value()}")
    
    print(f"   Final sample: {sampler.value()}")
    print(f"   Total items processed: {sampler.nobs()}")
    print()


def example_comparison_with_numpy():
    """Compare with numpy's built-in weighted choice."""
    print("=== Comparison with NumPy's weighted choice ===\n")
    
    # Set up test data
    rng = np.random.default_rng(42)
    elements = np.array([1, 2, 3, 4, 5])
    weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    
    num_trials = 10000
    
    # Our algorithm
    our_results = []
    for _ in range(num_trials):
        sample = weighted_sample_single(elements, weights, 
                                      np.random.default_rng(rng.integers(0, 2**32)))
        our_results.append(sample)
    
    # NumPy's choice
    numpy_results = rng.choice(elements, size=num_trials, p=weights)
    
    # Compare distributions
    print("Comparison of distributions:")
    print("Element | Our Alg | NumPy   | Expected")
    print("-" * 40)
    
    from collections import Counter
    our_counter = Counter(our_results)
    numpy_counter = Counter(numpy_results)
    
    for element, expected_prob in zip(elements, weights):
        our_freq = our_counter[element] / num_trials
        numpy_freq = numpy_counter[element] / num_trials
        print(f"{element:7} | {our_freq:7.3f} | {numpy_freq:7.3f} | {expected_prob:8.3f}")
    
    print("\nNote: Our algorithm is designed for streaming scenarios where")
    print("you don't know all elements in advance, unlike numpy.choice.")
    print()


def example_large_scale():
    """Example with larger scale data."""
    print("=== Large Scale Example ===\n")
    
    # Generate large dataset
    n_items = 100000
    rng = np.random.default_rng(42)
    
    # Create items with power-law weights (realistic scenario)
    items = np.arange(n_items)
    weights = 1.0 / (items + 1)  # Power law: 1/x
    weights = weights / weights.sum()  # Normalize
    
    print(f"Sampling from {n_items} items with power-law weights...")
    
    # Time the sampling
    import time
    
    start_time = time.time()
    sample = weighted_sample_single(items, weights, rng)
    end_time = time.time()
    
    print(f"Sample: {sample}")
    print(f"Weight of sampled item: {weights[sample]:.6f}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print()
    
    # Show that smaller indices (higher weights) are more likely
    print("Distribution check with multiple samples:")
    samples = []
    for _ in range(1000):
        sample = weighted_sample_single(items, weights, 
                                      np.random.default_rng(rng.integers(0, 2**32)))
        samples.append(sample)
    
    samples = np.array(samples)
    print(f"Mean sampled index: {samples.mean():.1f} (lower is better due to power-law)")
    print(f"Median sampled index: {np.median(samples):.1f}")
    print(f"Min sampled index: {samples.min()}")
    print(f"Max sampled index: {samples.max()}")
    print()


def example_practical_applications():
    """Show practical applications of the algorithm."""
    print("=== Practical Applications ===\n")
    
    # Application 1: Log sampling based on severity
    print("1. Log Entry Sampling (by severity):")
    log_entries = [
        ("DEBUG: Connection established", 1),
        ("INFO: User logged in", 2),
        ("WARNING: High memory usage", 5),
        ("ERROR: Database connection failed", 10),
        ("CRITICAL: System crash detected", 20)
    ]
    
    # Sample based on severity (higher severity = higher weight)
    messages, severities = zip(*log_entries)
    sampler = stream_weighted_sample_single(np.random.default_rng(42))
    
    print("   Processing log entries:")
    for message, severity in log_entries:
        sampler.fit(message, severity)
        print(f"   {message} -> Current sample: {sampler.value()}")
    
    print()
    
    # Application 2: Feature selection based on importance scores
    print("2. Feature Selection (by importance):")
    features = ['age', 'income', 'education', 'location', 'job_title']
    importance_scores = [0.8, 0.9, 0.6, 0.3, 0.7]
    
    selected_feature = weighted_sample_single(features, importance_scores)
    print(f"   Features: {features}")
    print(f"   Importance scores: {importance_scores}")
    print(f"   Selected feature: {selected_feature}")
    print()
    
    # Application 3: Weighted customer sampling for surveys
    print("3. Customer Sampling (by value):")
    customers = ['Customer_A', 'Customer_B', 'Customer_C', 'Customer_D']
    customer_values = [1000, 5000, 800, 3000]  # Customer lifetime value
    
    selected_customer = weighted_sample_single(customers, customer_values)
    print(f"   Customers: {customers}")
    print(f"   Lifetime values: {customer_values}")
    print(f"   Selected for survey: {selected_customer}")
    print()


def run_all_examples():
    """Run all example functions."""
    example_basic_usage()
    example_streaming()
    example_comparison_with_numpy()
    example_large_scale()
    example_practical_applications()


if __name__ == "__main__":
    run_all_examples()
