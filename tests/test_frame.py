import os

import pytest
import torch

try:
    from torch_frame import stype
except Exception:
    stype = None

from torchsky.frame import read_tensor_frame, to_tensor_frame, write_tensor_frame


def test_frame_integration():
    if stype is None:
        pytest.skip("torch_frame not installed")
    filename = "test_frame.fits"

    # Create sample data
    num_rows = 100
    data = {
        "col_num": torch.randn(num_rows),
        "col_cat": torch.randint(0, 5, (num_rows,), dtype=torch.int64),
    }

    # Convert to TensorFrame
    tf = to_tensor_frame(data)
    assert stype.numerical in tf.feat_dict
    assert stype.categorical in tf.feat_dict
    assert tf.feat_dict[stype.numerical].shape == (num_rows, 1)
    assert tf.feat_dict[stype.categorical].shape == (num_rows, 1)

    # Write to FITS
    write_tensor_frame(filename, tf, overwrite=True)
    assert os.path.exists(filename)
    # Read back
    tf_read = read_tensor_frame(filename)

    # Verify content
    # Note: FITS might change column order or names case (though torchsky tries to preserve)
    # torchsky write uses original names.

    # Check numerical
    assert stype.numerical in tf_read.feat_dict
    val_orig = tf.feat_dict[stype.numerical]
    val_read = tf_read.feat_dict[stype.numerical]
    # FITS float32 precision might cause small diffs if original was float64?
    # torch.randn is float32 by default in PyTorch? No, float32.
    assert torch.allclose(val_orig, val_read, atol=1e-5)

    # Check categorical
    assert stype.categorical in tf_read.feat_dict
    cat_orig = tf.feat_dict[stype.categorical]
    cat_read = tf_read.feat_dict[stype.categorical]
    assert torch.equal(cat_orig, cat_read)

    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)


def test_schema_override():
    if stype is None:
        pytest.skip("torch_frame not installed")

    num_rows = 10
    data = {
        "col_int_as_num": torch.randint(0, 5, (num_rows,), dtype=torch.int32),
        "col_int_as_cat": torch.randint(0, 5, (num_rows,), dtype=torch.int32),
        "col_float_as_cat": torch.randn(num_rows, dtype=torch.float32),
    }

    schema = {
        "col_int_as_cat": stype.categorical,
        "col_float_as_cat": stype.categorical,
    }

    tf = to_tensor_frame(data, schema=schema)

    assert stype.numerical in tf.feat_dict
    assert "col_int_as_num" in tf.col_names_dict[stype.numerical]

    assert stype.categorical in tf.feat_dict
    assert "col_int_as_cat" in tf.col_names_dict[stype.categorical]
    assert "col_float_as_cat" in tf.col_names_dict[stype.categorical]

    assert tf.feat_dict[stype.numerical].dtype == torch.float32
    assert tf.feat_dict[stype.categorical].dtype == torch.int64
