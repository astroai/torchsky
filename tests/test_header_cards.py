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

    assert header["OBJECT"] == "M33"
    assert header.card("OBJECT") == Card("OBJECT", "M33", "retargeted")
    assert [card.key for card in header.cards] == ["OBJECT", "EXPTIME"]
    assert header.comments("OBJECT") == ["retargeted"]


def test_header_history_comment_preserve_repeated_cards() -> None:
    header = Header()
    header.add_history("created")
    header.add_history("calibrated")
    header.add_comment("first")
    header.add_comment("second")

    assert header["HISTORY"] == "calibrated"
    assert header["COMMENT"] == "second"
    assert header.get_history() == ["created", "calibrated"]
    assert header.get_comment() == ["first", "second"]
    assert [card.value for card in header.cards if card.key == "HISTORY"] == [
        "created",
        "calibrated",
    ]


def test_header_card_insert_remove_and_copy() -> None:
    header = Header([("A", 1, "one"), Card("B", 2, "two")])
    header.insert(1, ("C", 3, "three"))
    header.append(("HISTORY", "done", ""))

    assert [card.key for card in header.cards] == ["A", "C", "B", "HISTORY"]
    assert header["C"] == 3
    assert header.get_history() == ["done"]

    copied = Header(header)
    assert copied.cards == header.cards
    assert dict(copied) == dict(header)

    header.remove("C")
    assert "C" not in header
    assert [card.key for card in header.cards] == ["A", "B", "HISTORY"]


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

    torchsky.write(str(path), torch.arange(4, dtype=torch.float32).reshape(2, 2), header=header)

    out = torchsky.get_header(str(path), 0)
    assert out["OBJECT"] == "M31"
    assert out.card("OBJECT").comment == "target name"
    assert out["FLAG"] is True or out["FLAG"] == "T"
    assert "created by torchsky" in out.get_history()
    assert "image comment" in out.get_comment()


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
    assert out["EXTNAME"] == "CAT"
    assert out.card("EXTNAME").comment == "table extension"
    assert "table history" in out.get_history()
