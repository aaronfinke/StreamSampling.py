"""
Extended examples and comparison for the Weighted Reservoir Sampling Multi-element implementation.

This file demonstrates various use cases and provides detailed comparisons between
the three implemented algorithms: A-Res, A-ExpJ, and WRSWR-SKIP.
"""

import numpy as np
from collections import Counter
from weighted_sampling_multi_python import (
    weighted_sample_multi, 
    stream_weighted_sample_multi,
    SamplingMethod
)


def example_basic_usage():
    """Basic usage examples for all three algorithms."""
    print("=== Basic Usage Examples ===\n")
    
    # Example data
    elements = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape']
    weights = [0.1, 0.2, 0.3, 0.15, 0.1, 0.1, 0.05]  # Cherry most likely
    n_sample = 3
    
    methods = [
        (SamplingMethod.A_RES, "Algorithm A-Res"),
        (SamplingMethod.A_EXPJ, "Algorithm A-ExpJ"), 
        (SamplingMethod.WRSWR_SKIP, "WRSWR-SKIP")
    ]
    
    print(f"Sampling {n_sample} elements from: {elements}")
    print(f"Weights: {weights}")
    print()
    
    for method, method_name in methods:
        sample = weighted_sample_multi(elements, weights, n_sample, method,
                                     rng=np.random.default_rng(42))
        print(f"{method_name}: {sample}")
    print()


def example_algorithm_comparison():
    """Compare the three algorithms with statistical analysis."""
    print("=== Algorithm Comparison ===\n")
    
    # Test with different scenarios
    scenarios = [
        ("Uniform weights", [1.0] * 10, "Equal probability for all elements"),
        ("Linear weights", list(range(1, 11)), "Probability proportional to element value"),
        ("Exponential weights", [2**i for i in range(10)], "Exponentially increasing weights"),
        ("Bimodal weights", [1, 1, 1, 1, 1, 10, 10, 10, 10, 10], "Two weight groups")
    ]
    
    n_sample = 3
    n_trials = 2000
    elements = list(range(1, 11))
    
    for scenario_name, weights, description in scenarios:
        print(f"Scenario: {scenario_name}")
        print(f"Description: {description}")
        print(f"Weights: {weights}")
        print()
        
        methods = [
            (SamplingMethod.A_RES, "A-Res"),
            (SamplingMethod.A_EXPJ, "A-ExpJ"),
            (SamplingMethod.WRSWR_SKIP, "WRSWR-SKIP")
        ]
        
        for method, method_name in methods:
            results = []
            rng = np.random.default_rng(42)
            
            for _ in range(n_trials):
                sample = weighted_sample_multi(elements, weights, n_sample, method,
                                             rng=np.random.default_rng(rng.integers(0, 2**32)))
                results.extend(sample)
            
            counter = Counter(results)
            total = sum(counter.values())
            total_weight = sum(weights)
            
            print(f"  {method_name} results:")
            print("    Element | Frequency | Expected | Difference")
            print("    " + "-" * 42)
            
            for i, element in enumerate(elements):
                count = counter.get(element, 0)
                frequency = count / total if total > 0 else 0
                expected = weights[i] / total_weight
                difference = abs(frequency - expected)
                print(f"    {element:7} | {frequency:9.3f} | {expected:8.3f} | {difference:10.3f}")
            print()
    

def example_streaming_comparison():
    """Compare algorithms in streaming scenarios."""
    print("=== Streaming Performance Comparison ===\n")
    
    # Create a stream with varying weights
    rng = np.random.default_rng(42)
    stream_size = 10000
    n_sample = 100
    
    # Generate stream with time-varying weights (simulating changing preferences)
    stream_data = []
    for i in range(stream_size):
        element = f"item_{i}"
        # Weight varies sinusoidally with some noise
        base_weight = 1.0 + 0.5 * np.sin(2 * np.pi * i / 1000)
        weight = max(0.1, base_weight + rng.normal(0, 0.1))
        stream_data.append((element, weight))
    
    methods = [
        (SamplingMethod.A_RES, "A-Res"),
        (SamplingMethod.A_EXPJ, "A-ExpJ"),
        (SamplingMethod.WRSWR_SKIP, "WRSWR-SKIP")
    ]
    
    print(f"Processing stream of {stream_size} elements, sampling {n_sample}:")
    print()
    
    import time
    
    for method, method_name in methods:
        sampler = stream_weighted_sample_multi(n_sample, method, rng=np.random.default_rng(42))
        
        start_time = time.time()
        for element, weight in stream_data:
            sampler.fit(element, weight)
        end_time = time.time()
        
        final_sample = sampler.value()
        
        print(f"{method_name}:")
        print(f"  Processing time: {end_time - start_time:.4f} seconds")
        print(f"  Final sample size: {len(final_sample)}")
        print(f"  Elements processed: {sampler.nobs()}")
        
        # Check distribution of selected indices (should favor later elements due to higher weights)
        indices = [int(item.split('_')[1]) for item in final_sample if 'item_' in item]
        if indices:
            print(f"  Mean selected index: {np.mean(indices):.1f} (out of {stream_size-1})")
            print(f"  Selected index range: {min(indices)} - {max(indices)}")
        print()


def example_ordered_vs_unordered():
    """Demonstrate ordered vs unordered sampling."""
    print("=== Ordered vs Unordered Sampling ===\n")
    
    elements = [f"task_{i:02d}" for i in range(20)]
    # Weights favor earlier tasks
    weights = [20 - i for i in range(20)]
    n_sample = 5
    
    print(f"Elements: {elements[:10]}... (20 total)")
    print(f"Weights: {weights[:10]}... (decreasing)")
    print(f"Sample size: {n_sample}")
    print()
    
    rng = np.random.default_rng(42)
    
    # Test with different algorithms
    methods = [
        (SamplingMethod.A_RES, "A-Res"),
        (SamplingMethod.A_EXPJ, "A-ExpJ"),
        (SamplingMethod.WRSWR_SKIP, "WRSWR-SKIP")
    ]
    
    for method, method_name in methods:
        # Unordered sampling
        sample_unordered = weighted_sample_multi(elements, weights, n_sample, method, 
                                               ordered=False, rng=np.random.default_rng(rng.integers(0, 2**32)))
        
        # Ordered sampling
        sample_ordered = weighted_sample_multi(elements, weights, n_sample, method, 
                                             ordered=True, rng=np.random.default_rng(rng.integers(0, 2**32)))
        
        print(f"{method_name}:")
        print(f"  Unordered: {sample_unordered}")
        print(f"  Ordered:   {sample_ordered}")
        print()


def example_weight_functions():
    """Examples using weight functions instead of weight arrays."""
    print("=== Using Weight Functions ===\n")
    
    # Example 1: Text analysis - weight by string length
    documents = [
        "short",
        "a bit longer document",
        "this is a medium length document with some content",
        "very short",
        "this document is quite long and contains much more text than the others",
        "medium doc",
        "another relatively long document that has substantial content",
        "tiny"
    ]
    
    def length_weight(doc):
        """Weight by document length."""
        return len(doc)
    
    print("Example 1: Document sampling by length")
    print("Documents:", [f'"{doc[:20]}{"..." if len(doc) > 20 else ""}"' for doc in documents])
    print()
    
    sample = weighted_sample_multi(documents, length_weight, n=3, 
                                 method=SamplingMethod.A_EXPJ,
                                 rng=np.random.default_rng(42))
    
    print("Selected documents:")
    for doc in sample:
        print(f'  "{doc}" (length: {len(doc)})')
    print()
    
    # Example 2: Priority queue simulation
    print("Example 2: Task priority simulation")
    tasks = [
        {"id": 1, "priority": "high", "complexity": 3},
        {"id": 2, "priority": "low", "complexity": 1},
        {"id": 3, "priority": "medium", "complexity": 2},
        {"id": 4, "priority": "high", "complexity": 1},
        {"id": 5, "priority": "low", "complexity": 3},
        {"id": 6, "priority": "medium", "complexity": 3},
        {"id": 7, "priority": "high", "complexity": 2},
    ]
    
    def task_weight(task):
        """Weight by priority and inverse complexity."""
        priority_weights = {"low": 1, "medium": 3, "high": 5}
        return priority_weights[task["priority"]] / task["complexity"]
    
    selected_tasks = weighted_sample_multi(tasks, task_weight, n=3,
                                         method=SamplingMethod.A_RES,
                                         rng=np.random.default_rng(42))
    
    print("Selected tasks:")
    for task in selected_tasks:
        weight = task_weight(task)
        print(f"  Task {task['id']}: {task['priority']} priority, "
              f"complexity {task['complexity']}, weight {weight:.2f}")
    print()


def example_practical_applications():
    """Show practical applications of weighted multi-element sampling."""
    print("=== Practical Applications ===\n")
    
    # Application 1: A/B testing with weighted groups
    print("Application 1: A/B Testing with Weighted Groups")
    
    test_groups = [
        {"name": "control", "capacity": 1000, "priority": 1.0},
        {"name": "variant_a", "capacity": 500, "priority": 1.5},
        {"name": "variant_b", "capacity": 300, "priority": 2.0},
        {"name": "variant_c", "capacity": 200, "priority": 1.2},
    ]
    
    # Weight by priority and capacity
    def group_weight(group):
        return group["priority"] * np.sqrt(group["capacity"])
    
    # Select 2 groups for comparison
    selected_groups = weighted_sample_multi(test_groups, group_weight, n=2,
                                          method=SamplingMethod.A_EXPJ,
                                          rng=np.random.default_rng(42))
    
    print("Selected test groups:")
    for group in selected_groups:
        weight = group_weight(group)
        print(f"  {group['name']}: capacity={group['capacity']}, "
              f"priority={group['priority']}, weight={weight:.2f}")
    print()
    
    # Application 2: Feature selection for machine learning
    print("Application 2: Feature Selection")
    
    features = [
        {"name": "age", "importance": 0.8, "correlation": 0.1},
        {"name": "income", "importance": 0.9, "correlation": 0.3},
        {"name": "education", "importance": 0.6, "correlation": 0.2},
        {"name": "location", "importance": 0.4, "correlation": 0.1},
        {"name": "job_title", "importance": 0.7, "correlation": 0.4},
        {"name": "experience", "importance": 0.8, "correlation": 0.5},
        {"name": "skills", "importance": 0.9, "correlation": 0.2},
    ]
    
    def feature_weight(feature):
        """Weight by importance, penalized by correlation with other features."""
        return feature["importance"] / (1 + feature["correlation"])
    
    selected_features = weighted_sample_multi(features, feature_weight, n=4,
                                            method=SamplingMethod.WRSWR_SKIP,
                                            rng=np.random.default_rng(42))
    
    print("Selected features:")
    for feature in selected_features:
        weight = feature_weight(feature)
        print(f"  {feature['name']}: importance={feature['importance']}, "
              f"correlation={feature['correlation']}, weight={weight:.2f}")
    print()


def visualize_algorithm_differences():
    """Create visualizations comparing the algorithms (requires matplotlib)."""
    print("=== Algorithm Visualization ===\n")
    
    try:
        import matplotlib.pyplot as plt
        
        # Generate data for comparison
        elements = list(range(1, 21))
        weights = [i**0.5 for i in elements]  # Square root weights
        n_sample = 5
        n_trials = 1000
        
        methods = [
            (SamplingMethod.A_RES, "A-Res"),
            (SamplingMethod.A_EXPJ, "A-ExpJ"),
            (SamplingMethod.WRSWR_SKIP, "WRSWR-SKIP")
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Algorithm Comparison: Element Selection Frequencies')
        
        for idx, (method, method_name) in enumerate(methods):
            results = []
            rng = np.random.default_rng(42)
            
            for _ in range(n_trials):
                sample = weighted_sample_multi(elements, weights, n_sample, method,
                                             rng=np.random.default_rng(rng.integers(0, 2**32)))
                results.extend(sample)
            
            counter = Counter(results)
            freqs = [counter.get(elem, 0) / len(results) for elem in elements]
            expected = np.array(weights) / sum(weights)
            
            ax = axes[idx]
            x = np.arange(len(elements))
            width = 0.35
            
            ax.bar(x - width/2, freqs, width, label='Observed', alpha=0.7)
            ax.bar(x + width/2, expected, width, label='Expected', alpha=0.7)
            
            ax.set_title(method_name)
            ax.set_xlabel('Element')
            ax.set_ylabel('Frequency')
            ax.legend()
            # Show every other element label to avoid crowding
            ax.set_xticks(x[::2])
            ax.set_xticklabels([str(e) for e in elements[::2]], rotation=45)
        
        plt.tight_layout()
        plt.savefig('/Users/aaronfinke/StreamSampling.jl/algorithm_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Visualization saved as 'algorithm_comparison.png'")
        print("The plot shows observed vs expected frequencies for each algorithm.")
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")
    
    print()


def run_all_examples():
    """Run all example functions."""
    example_basic_usage()
    example_algorithm_comparison()
    example_streaming_comparison()
    example_ordered_vs_unordered()
    example_weight_functions()
    example_practical_applications()
    visualize_algorithm_differences()


if __name__ == "__main__":
    run_all_examples()
