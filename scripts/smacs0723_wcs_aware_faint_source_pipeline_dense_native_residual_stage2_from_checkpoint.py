#!/usr/bin/env python3
"""Run N/F/P/J Stage-2 ablations from one dense Stage-1 checkpoint.

This launcher is built on
``smacs0723_wcs_aware_faint_source_pipeline_dense_native_residual.py`` and keeps
its dense native residual, rendering, checkpoint, and detector behavior.

Branches
--------
N: no-PSF continuation; update scene/background