"""
Performance and benchmark tests for AI Safety Guardrails.
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

from ai_safety_guardrails import SafetyGuard
from ai_safety_guardrails.detectors.base_detector import DetectionResult


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for the safety system."""

    @pytest.mark.asyncio
    async def test_single_detector_performance(self, mock_detector, benchmark_test_cases):
        """Benchmark single detector performance."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            # Warm up
            await guard.analyze_text("warm up")
            
            # Benchmark
            times = []
            for case in benchmark_test_cases:
                start_time = time.time()
                await guard.analyze_text(case["text"])
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Performance assertions
            avg_time = statistics.mean(times)
            max_time = max(times)
            min_time = min(times)
            
            assert avg_time < 0.1, f"Average time {avg_time:.3f}s too slow"
            assert max_time < 0.5, f"Max time {max_time:.3f}s too slow"
            assert min_time > 0, "Minimum time should be positive"
            
            print(f"Single detector performance:")
            print(f"  Average: {avg_time:.3f}s")
            print(f"  Min: {min_time:.3f}s")
            print(f"  Max: {max_time:.3f}s")
            print(f"  Std Dev: {statistics.stdev(times):.3f}s")

    @pytest.mark.asyncio
    async def test_multiple_detectors_performance(self, mock_detector, benchmark_test_cases):
        """Benchmark multiple detectors performance."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            # Test with all detectors
            guard = SafetyGuard(detectors=["toxicity", "pii", "prompt_injection", "spam", "topics"])
            
            # Warm up
            await guard.analyze_text("warm up")
            
            # Benchmark
            times = []
            for case in benchmark_test_cases:
                start_time = time.time()
                await guard.analyze_text(case["text"])
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Performance assertions
            avg_time = statistics.mean(times)
            max_time = max(times)
            
            # Multiple detectors should still be reasonably fast
            assert avg_time < 0.5, f"Average time {avg_time:.3f}s too slow for multiple detectors"
            assert max_time < 1.0, f"Max time {max_time:.3f}s too slow for multiple detectors"
            
            print(f"Multiple detectors performance:")
            print(f"  Average: {avg_time:.3f}s")
            print(f"  Min: {min(times):.3f}s")
            print(f"  Max: {max_time:.3f}s")

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, mock_detector, sample_texts):
        """Benchmark batch processing performance."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            # Test batch sizes
            batch_sizes = [1, 5, 10, 20]
            results = {}
            
            for batch_size in batch_sizes:
                texts = sample_texts["safe"][:batch_size]
                
                start_time = time.time()
                await guard.batch_analyze(texts)
                end_time = time.time()
                
                total_time = end_time - start_time
                time_per_text = total_time / batch_size
                
                results[batch_size] = {
                    "total_time": total_time,
                    "time_per_text": time_per_text
                }
                
                print(f"Batch size {batch_size}: {time_per_text:.3f}s per text")
            
            # Batch processing should be more efficient than individual calls
            individual_time = results[1]["time_per_text"]
            batch_time = results[10]["time_per_text"] if 10 in results else results[max(results.keys())]["time_per_text"]
            
            # Batch should be at least 20% more efficient
            efficiency_ratio = individual_time / batch_time
            assert efficiency_ratio >= 1.2, f"Batch processing not efficient enough: {efficiency_ratio:.2f}x"

    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self, mock_detector, sample_texts):
        """Benchmark concurrent processing performance."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"], max_concurrent_detections=5)
            
            # Test concurrent vs sequential
            texts = sample_texts["safe"][:10]
            
            # Sequential processing
            start_time = time.time()
            for text in texts:
                await guard.analyze_text(text)
            sequential_time = time.time() - start_time
            
            # Concurrent processing
            start_time = time.time()
            tasks = [guard.analyze_text(text) for text in texts]
            await asyncio.gather(*tasks)
            concurrent_time = time.time() - start_time
            
            # Concurrent should be faster
            speedup = sequential_time / concurrent_time
            assert speedup > 1.5, f"Concurrent processing not fast enough: {speedup:.2f}x speedup"
            
            print(f"Concurrency performance:")
            print(f"  Sequential: {sequential_time:.3f}s")
            print(f"  Concurrent: {concurrent_time:.3f}s")
            print(f"  Speedup: {speedup:.2f}x")

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, mock_detector, sample_texts):
        """Test memory usage stability during extended operation."""
        import psutil
        import gc
        
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            # Baseline memory
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_readings = [baseline_memory]
            
            # Process many texts
            for i in range(100):
                text = sample_texts["safe"][i % len(sample_texts["safe"])]
                await guard.analyze_text(text)
                
                if i % 20 == 0:  # Sample memory every 20 iterations
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_readings.append(current_memory)
            
            # Force garbage collection
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_readings.append(final_memory)
            
            # Memory should not grow excessively
            memory_growth = final_memory - baseline_memory
            max_memory = max(memory_readings)
            memory_peak = max_memory - baseline_memory
            
            assert memory_growth < 100, f"Memory grew too much: {memory_growth:.1f}MB"
            assert memory_peak < 200, f"Memory peak too high: {memory_peak:.1f}MB"
            
            print(f"Memory usage:")
            print(f"  Baseline: {baseline_memory:.1f}MB")
            print(f"  Peak: {max_memory:.1f}MB (+{memory_peak:.1f}MB)")
            print(f"  Final: {final_memory:.1f}MB (+{memory_growth:.1f}MB)")


@pytest.mark.slow
class TestScalabilityTests:
    """Test system scalability under load."""

    @pytest.mark.asyncio
    async def test_high_concurrency_load(self, mock_detector):
        """Test system under high concurrency load."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            # Add small delay to simulate real processing
            async def slow_detect(text):
                await asyncio.sleep(0.01)  # 10ms processing time
                return DetectionResult(
                    blocked=False,
                    confidence=0.1,
                    reason=None,
                    detector_name="mock_detector",
                    processing_time=0.01
                )
            
            mock_detector_instance = mock_detector(should_block=False)
            mock_detector_instance.safe_detect = slow_detect
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"], max_concurrent_detections=20)
            
            # Create many concurrent requests
            num_requests = 100
            texts = [f"Test message {i}" for i in range(num_requests)]
            
            start_time = time.time()
            tasks = [guard.analyze_text(text) for text in texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Check results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]
            
            total_time = end_time - start_time
            throughput = len(successful_results) / total_time
            
            # Should handle high load successfully
            success_rate = len(successful_results) / num_requests
            assert success_rate >= 0.95, f"Success rate too low: {success_rate:.2%}"
            assert throughput > 10, f"Throughput too low: {throughput:.1f} req/s"
            
            print(f"High concurrency load test:")
            print(f"  Requests: {num_requests}")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Throughput: {throughput:.1f} req/s")
            print(f"  Failed requests: {len(failed_results)}")

    @pytest.mark.asyncio
    async def test_stress_test_extended_operation(self, mock_detector):
        """Stress test with extended operation."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity", "pii"])
            
            # Run for extended period
            start_time = time.time()
            error_count = 0
            success_count = 0
            
            # Run for 30 seconds or 1000 requests, whichever comes first
            while time.time() - start_time < 30 and success_count < 1000:
                try:
                    text = f"Stress test message {success_count}"
                    await guard.analyze_text(text)
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    if error_count > 10:  # Too many errors
                        break
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)
            
            total_time = time.time() - start_time
            error_rate = error_count / (success_count + error_count)
            
            # System should remain stable
            assert error_rate < 0.01, f"Error rate too high: {error_rate:.2%}"
            assert success_count > 100, f"Not enough successful requests: {success_count}"
            
            print(f"Stress test results:")
            print(f"  Duration: {total_time:.1f}s")
            print(f"  Successful requests: {success_count}")
            print(f"  Failed requests: {error_count}")
            print(f"  Error rate: {error_rate:.2%}")
            print(f"  Average rate: {success_count/total_time:.1f} req/s")

    @pytest.mark.asyncio
    async def test_resource_cleanup_under_load(self, mock_detector):
        """Test resource cleanup under high load."""
        import gc
        import weakref
        
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            # Track guard instances
            guard_refs = []
            
            for i in range(50):  # Create many guard instances
                guard = SafetyGuard(detectors=["toxicity"])
                guard_refs.append(weakref.ref(guard))
                
                # Use the guard briefly
                await guard.analyze_text(f"Test {i}")
                
                # Delete reference
                del guard
                
                # Periodic cleanup
                if i % 10 == 0:
                    gc.collect()
            
            # Force final cleanup
            gc.collect()
            await asyncio.sleep(0.1)  # Allow cleanup to complete
            
            # Check how many instances were cleaned up
            alive_count = sum(1 for ref in guard_refs if ref() is not None)
            cleanup_rate = (len(guard_refs) - alive_count) / len(guard_refs)
            
            # Most instances should be cleaned up
            assert cleanup_rate > 0.8, f"Cleanup rate too low: {cleanup_rate:.2%}"
            
            print(f"Resource cleanup test:")
            print(f"  Created instances: {len(guard_refs)}")
            print(f"  Cleaned up: {len(guard_refs) - alive_count}")
            print(f"  Cleanup rate: {cleanup_rate:.2%}")


@pytest.mark.slow
class TestPerformanceRegression:
    """Performance regression tests."""

    @pytest.mark.asyncio
    async def test_performance_regression_baseline(self, mock_detector, benchmark_test_cases):
        """Establish performance baseline for regression testing."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            # Baseline measurements
            times = []
            for case in benchmark_test_cases:
                start_time = time.time()
                await guard.analyze_text(case["text"])
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            p95_time = sorted(times)[int(len(times) * 0.95)]
            
            # Store baseline for comparison (in real tests, this would be stored persistently)
            baseline = {
                "avg_time": avg_time,
                "p95_time": p95_time,
                "detector_count": 1,
                "test_cases": len(benchmark_test_cases)
            }
            
            print(f"Performance baseline:")
            print(f"  Average time: {avg_time:.3f}s")
            print(f"  P95 time: {p95_time:.3f}s")
            print(f"  Test cases: {len(benchmark_test_cases)}")
            
            # Basic sanity checks
            assert avg_time < 0.1, "Baseline performance too slow"
            assert p95_time < 0.2, "P95 performance too slow"

    @pytest.mark.asyncio
    async def test_accuracy_vs_performance_tradeoff(self, mock_detector):
        """Test the tradeoff between accuracy and performance."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            # Different detector configurations with varying complexity
            configs = [
                {"detectors": ["toxicity"], "name": "single"},
                {"detectors": ["toxicity", "pii"], "name": "dual"},
                {"detectors": ["toxicity", "pii", "prompt_injection"], "name": "triple"},
                {"detectors": ["toxicity", "pii", "prompt_injection", "spam", "topics"], "name": "full"},
            ]
            
            results = {}
            
            for config in configs:
                mock_detector_instance = mock_detector(should_block=False)
                mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
                
                guard = SafetyGuard(detectors=config["detectors"])
                
                # Measure performance
                times = []
                for i in range(20):
                    start_time = time.time()
                    await guard.analyze_text(f"Test message {i}")
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = statistics.mean(times)
                results[config["name"]] = {
                    "avg_time": avg_time,
                    "detector_count": len(config["detectors"])
                }
                
                print(f"{config['name']} config: {avg_time:.3f}s avg ({len(config['detectors'])} detectors)")
            
            # Performance should scale reasonably with detector count
            single_time = results["single"]["avg_time"]
            full_time = results["full"]["avg_time"]
            scaling_factor = full_time / single_time
            
            # Full config should not be more than 5x slower than single
            assert scaling_factor < 5.0, f"Performance scaling too poor: {scaling_factor:.1f}x"


class TestProfilerIntegration:
    """Integration with profiling tools for performance analysis."""

    @pytest.mark.asyncio
    async def test_cprofile_integration(self, mock_detector, sample_texts):
        """Test with cProfile for detailed performance analysis."""
        import cProfile
        import pstats
        import io
        
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            # Profile the operation
            profiler = cProfile.Profile()
            profiler.enable()
            
            # Run operations
            for text in sample_texts["safe"][:10]:
                await guard.analyze_text(text)
            
            profiler.disable()
            
            # Analyze results
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(10)  # Top 10 functions
            
            profile_output = stats_stream.getvalue()
            
            # Basic checks on profile data
            assert "analyze_text" in profile_output
            assert len(profile_output) > 0
            
            print("Top performance hotspots:")
            print(profile_output)

    @pytest.mark.asyncio
    async def test_memory_profiling(self, mock_detector, sample_texts):
        """Test memory profiling during operation."""
        try:
            import tracemalloc
        except ImportError:
            pytest.skip("tracemalloc not available")
        
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            # Start memory tracing
            tracemalloc.start()
            
            # Run operations
            for text in sample_texts["safe"][:50]:
                await guard.analyze_text(text)
            
            # Get memory statistics
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            current_mb = current / 1024 / 1024
            peak_mb = peak / 1024 / 1024
            
            print(f"Memory tracing:")
            print(f"  Current: {current_mb:.1f}MB")
            print(f"  Peak: {peak_mb:.1f}MB")
            
            # Memory usage should be reasonable
            assert peak_mb < 50, f"Peak memory too high: {peak_mb:.1f}MB"