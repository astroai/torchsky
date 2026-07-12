"""FITS header parser compatibility wrapper over torchfits 0.8."""

import time
from typing import Any

from torchfits.header_parser import fast_parse_header as fast_parse_header


class FastHeaderParser:
    """Small timing wrapper around torchfits' public header parser."""

    @staticmethod
    def parse_header_string(header_string: str) -> dict[str, Any]:
        return fast_parse_header(header_string)

    @staticmethod
    def parse_with_performance_tracking(
        header_string: str,
    ) -> tuple[dict[str, Any], dict[str, float | int]]:
        start = time.perf_counter()
        parsed = fast_parse_header(header_string)
        elapsed = time.perf_counter() - start
        elapsed_ms = elapsed * 1000.0
        size = len(header_string)
        return parsed, {
            "header_size_bytes": size,
            "num_keywords": len(parsed),
            "parse_time_ms": elapsed_ms,
            "throughput_kb_per_ms": (size / 1024.0 / elapsed_ms) if elapsed_ms else 0,
        }
