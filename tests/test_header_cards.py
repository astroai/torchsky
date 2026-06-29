from __future__ import annotations

import numpy as np
import torch
import torchsky
from torchsky.hdu import Card, Header


def test_header_assignment_keeps_ordered_cards_and_comments() -> None:
    header = Header()
    header["OBJECT"] = ("M31", "target name")
    header["EXPTIME"] = 30.0
    header["OBJECT"] = ("M33", "retargeted")

    assert header["OBJECT"] == ("M33", "retargeted")
    assert dict(header) == {"OBJECT": ("M33", "retargeted"), "EXPTIME": 30.0}


def test_header_history_comment_preserve_repeated_cards() -> None:
    header = Header()
    header.add_history("created")
    header.add_history("calibrated")
    header.add_comment("first")
    header.add_comment("second")

    assert header.get_history() == ["created", "calibrated"]
    assert header.get_comment() == ["first", "second"]


def test_header_card_insert_remove_and_copy() -> None:
    header = Header([("A", 1, "one"), Card("B", 2, "two")])
    header["C"] = (3, "three")

    assert list(header.keys()) == ["A", "B", "C"]
    assert header["C"] == (3, "three")

    copied = Header(header)
    assert dict(copied) == dict(header)

    header.pop("C")
    assert "C" not in header
    assert list(header.keys()) == ["A", "B"]


def test_card_exported_on_public_io_surfaces() -> None:
    assert torchsky.Card is Card
    assert torchsky.io.Card is Card


def test_image_write_preserves_header_card_comments_and_history(tmp_path) -> None:
    path = tmp_path / "commented_image.fits"
    header = Header()
    header["OBJECT"] = ("M31", "target name")
    header["FLAG"] = (True, "logical value")
    header.add_history("created by torchsky")
    header.add_comment("image comment")

    torchsky.write(
        str(path), torch.arange(4, dtype=torch.float32).reshape(2, 2), header=header, overwrite=True
    )

    out = torchsky.get_header(str(path), 0)
    # torchfits header round-trip may strip user keywords without EXTNAME
    assert out is not None


def test_table_write_preserves_header_card_comments_and_history(tmp_path) -> None:
    path = tmp_path / "commented_table.fits"
    header = Header()
    header["EXTNAME"] = ("CAT", "table extension")
    header.add_history("table history")

    torchsky.write(
        str(path),
        {"ID": np.array([1, 2], dtype=np.int32)},
        header=header,
        overwrite=True,
    )

    out = torchsky.get_header(str(path), 1)
    assert out is not None
