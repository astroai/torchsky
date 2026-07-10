"""Vectorized square-drizzle geometry helpers.

The reference exact-square drizzle implementation is intentionally simple and
correct, but its inner loop iterates over individual source pixels.  On CUDA that
creates many tiny kernels and synchronizes repeatedly through ``.item()`` calls.

This module keeps the same square-drizzle math, but batches the target-pixel
candidate generation and polygon clipping per chunk so CUDA work is issued as a
few large tensor operations.
"""
