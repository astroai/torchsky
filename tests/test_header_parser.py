# Mock out torchsky.cpp since we only care about header_parser.py testing right now

from torchsky.header_parser import FastHeaderParser
from typing import Dict


def benchmark_header_parsing(
    header_string: str, num_iterations: int = 100
) -> Dict[str, float]:
    """
    Benchmark header parsing performance.

    Args:
        header_string: Header string to parse
        num_iterations: Number of parsing iterations

    Returns:
        Performance metrics dictionary
    """
    import time

    # Warmup
    FastHeaderParser.parse_header_string(header_string)

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        FastHeaderParser.parse_header_string(header_string)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000
    throughput_headers_per_sec = num_iterations / total_time

    return {
        "avg_parse_time_ms": avg_time_ms,
        "throughput_headers_per_sec": throughput_headers_per_sec,
        "total_time_s": total_time,
        "header_size_bytes": len(header_string),
        "num_iterations": num_iterations,
    }


def test_benchmark_header_parsing():
    """Test the benchmark_header_parsing function."""
    # Create a simple valid FITS header string
    header_cards = [
        "SIMPLE  =                    T / file does conform to FITS standard             ",
        "BITPIX  =                    8 / number of bits per data pixel                  ",
        "NAXIS   =                    0 / number of data axes                            ",
        "EXTEND  =                    T / FITS dataset may contain extensions            ",
        "END                                                                             ",
    ]
    # Pad to exactly 80 characters per card
    header_cards = [card.ljust(80) for card in header_cards]
    header_string = "".join(header_cards)

    # Run benchmark
    num_iterations = 10
    metrics = benchmark_header_parsing(header_string, num_iterations=num_iterations)

    # Check metrics dictionary keys and types
    assert isinstance(metrics, dict)

    expected_keys = [
        "avg_parse_time_ms",
        "throughput_headers_per_sec",
        "total_time_s",
        "header_size_bytes",
        "num_iterations",
    ]

    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], (float, int))

    # Check specific values
    assert metrics["num_iterations"] == num_iterations
    assert metrics["header_size_bytes"] == len(header_string)
    assert metrics["avg_parse_time_ms"] >= 0
    assert metrics["throughput_headers_per_sec"] >= 0
    assert metrics["total_time_s"] >= 0


def test_parse_with_performance_tracking():
    """Test the parse_with_performance_tracking method."""
    from unittest import mock

    # Create a simple valid FITS header string
    header_cards = [
        "SIMPLE  =                    T / file does conform to FITS standard             ",
        "BITPIX  =                    8 / number of bits per data pixel                  ",
        "NAXIS   =                    0 / number of data axes                            ",
        "EXTEND  =                    T / FITS dataset may contain extensions            ",
        "END                                                                             ",
    ]
    # Pad to exactly 80 characters per card
    header_cards = [card.ljust(80) for card in header_cards]
    header_string = "".join(header_cards)

    # 1. Test normal execution
    result = FastHeaderParser.parse_with_performance_tracking(header_string)

    assert isinstance(result, tuple)
    assert len(result) == 2

    header_dict, metrics_dict = result

    # Check header_dict
    assert isinstance(header_dict, dict)
    assert "SIMPLE" in header_dict
    assert header_dict["SIMPLE"] is True
    assert "BITPIX" in header_dict
    assert header_dict["BITPIX"] == 8

    # Check metrics_dict keys
    assert isinstance(metrics_dict, dict)
    expected_keys = [
        "parse_time_ms",
        "header_size_bytes",
        "num_keywords",
        "throughput_kb_per_ms",
    ]
    for key in expected_keys:
        assert key in metrics_dict
        assert isinstance(metrics_dict[key], (int, float))

    # Check specific metrics values
    assert metrics_dict["header_size_bytes"] == len(header_string)
    assert metrics_dict["num_keywords"] == len(header_dict)
    assert metrics_dict["parse_time_ms"] >= 0

    # 2. Test divide-by-zero protection (end_time == start_time)
    with mock.patch("time.perf_counter", return_value=123.456):
        _, zero_time_metrics = FastHeaderParser.parse_with_performance_tracking(
            header_string
        )

        # When time difference is exactly 0, throughput should fallback to 0
        assert zero_time_metrics["parse_time_ms"] == 0.0
        assert zero_time_metrics["throughput_kb_per_ms"] == 0.0
