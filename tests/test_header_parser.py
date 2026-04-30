# Mock out torchsky.cpp since we only care about header_parser.py testing right now

from typing import Dict
from unittest import mock

from torchsky.header_parser import FastHeaderParser, fast_parse_header


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
    header_cards = [
        "SIMPLE  =                    T / file does conform to FITS standard             ",
        "BITPIX  =                    8 / number of bits per data pixel                  ",
        "NAXIS   =                    0 / number of data axes                            ",
        "EXTEND  =                    T / FITS dataset may contain extensions            ",
        "END                                                                             ",
    ]
    header_string = "".join(card.ljust(80) for card in header_cards)

    header, metrics = FastHeaderParser.parse_with_performance_tracking(header_string)

    assert header["SIMPLE"] is True
    assert header["BITPIX"] == 8
    assert metrics["header_size_bytes"] == len(header_string)
    assert metrics["num_keywords"] == len(header)
    assert metrics["parse_time_ms"] >= 0
    assert metrics["throughput_kb_per_ms"] >= 0


def test_parse_with_performance_tracking_handles_zero_elapsed_time():
    header_string = "END".ljust(80)

    with mock.patch("time.perf_counter", return_value=123.456):
        _, metrics = FastHeaderParser.parse_with_performance_tracking(header_string)

    assert metrics["parse_time_ms"] == 0.0
    assert metrics["throughput_kb_per_ms"] == 0


class TestFastParseHeader:
    """Coverage for the public fast_parse_header wrapper."""

    def test_happy_path(self):
        header_cards = [
            "SIMPLE  =                    T / file does conform to FITS standard             ",
            "BITPIX  =                    8 / number of bits per data pixel                  ",
            "NAXIS   =                    0 / number of data axes                            ",
            "OBJECT  = 'M31     '           / target object                                  ",
            "EXPTIME =                100.5 / exposure time in seconds                       ",
            "OBSERVER= 'John Doe'           / observer name                                  ",
            "END                                                                             ",
        ]
        parsed = fast_parse_header("".join(card.ljust(80) for card in header_cards))

        assert parsed["SIMPLE"] is True
        assert parsed["BITPIX"] == 8
        assert parsed["NAXIS"] == 0
        assert parsed["OBJECT"] == "M31     "
        assert parsed["EXPTIME"] == 100.5
        assert parsed["OBSERVER"] == "John Doe"

    def test_empty_string(self):
        assert fast_parse_header("") == {}

    def test_no_end_card(self):
        header_cards = [
            "SIMPLE  =                    T / file does conform to FITS standard             ",
            "BITPIX  =                    8 / number of bits per data pixel                  ",
        ]
        parsed = fast_parse_header("".join(card.ljust(80) for card in header_cards))

        assert parsed["SIMPLE"] is True
        assert parsed["BITPIX"] == 8

    def test_complex_value(self):
        header_cards = [
            "COMPLEX = (1.0, -2.5)          / complex number                                 ",
            "END                                                                             ",
        ]
        parsed = fast_parse_header("".join(card.ljust(80) for card in header_cards))

        assert parsed["COMPLEX"] == complex(1.0, -2.5)

    def test_comment_and_history_cards(self):
        header_cards = [
            "COMMENT This is just a comment                                                  ",
            "HISTORY This is a history record                                                ",
            "END                                                                             ",
        ]
        parsed = fast_parse_header("".join(card.ljust(80) for card in header_cards))

        assert parsed["COMMENT"] == "This is just a comment"
        assert parsed["HISTORY"] == "This is a history record"
