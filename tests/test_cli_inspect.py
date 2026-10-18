import json

import torch

from torchsky.cli import build_parser
import torchfits


def test_info_json_summarizes_hdus_without_reading_data(tmp_path, capsys) -> None:
    path = tmp_path / "sample.fits"
    torchfits.write(str(path), torch.arange(6, dtype=torch.float32).reshape(2, 3))
    torchfits.insert_hdu(
        str(path),
        {
            "RA": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            "DEC": torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float64),
        },
        header={"EXTNAME": "CAT"},
    )

    args = build_parser().parse_args(["info", str(path), "--json"])
    assert args.handler(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["path"] == str(path)
    assert payload["hdu_count"] == 2
    assert payload["hdus"][0]["kind"] == "primary_image"
    assert payload["hdus"][0]["shape"] == [2, 3]
    assert payload["hdus"][1]["kind"] == "table"
    assert payload["hdus"][1]["rows"] == 3
    assert payload["hdus"][1]["columns"] == ["RA", "DEC"]


def test_info_text_lists_hdu_summary(tmp_path, capsys) -> None:
    path = tmp_path / "image.fits"
    torchfits.write(str(path), torch.zeros(2, 2))

    args = build_parser().parse_args(["info", str(path)])
    assert args.handler(args) == 0

    out = capsys.readouterr().out
    assert "File:" in out
    assert "primary_image" in out
    assert "2x2" in out


def test_cat_json_reads_bounded_table_sample(tmp_path, capsys) -> None:
    path = tmp_path / "table.fits"
    torchfits.write(
        str(path),
        {
            "RA": torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64),
            "DEC": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            "MAG": torch.tensor([18.0, 19.0, 20.0], dtype=torch.float64),
        },
        header={"EXTNAME": "CAT"},
    )

    args = build_parser().parse_args(
        [
            "cat",
            str(path),
            "--hdu",
            "CAT",
            "--columns",
            "RA,DEC",
            "--rows",
            "2",
            "--json",
        ]
    )
    assert args.handler(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == ["RA", "DEC"]
    assert payload["rows"] == [
        {"RA": 10.0, "DEC": 1.0},
        {"RA": 20.0, "DEC": 2.0},
    ]


def test_cat_text_prints_tab_separated_rows(tmp_path, capsys) -> None:
    path = tmp_path / "table.fits"
    torchfits.write(
        str(path),
        {
            "ID": torch.tensor([1, 2], dtype=torch.int64),
            "FLUX": torch.tensor([5.5, 6.5], dtype=torch.float64),
        },
        header={"EXTNAME": "CAT"},
    )

    args = build_parser().parse_args(
        ["cat", str(path), "--hdu", "CAT", "--columns", "ID,FLUX", "--rows", "1"]
    )
    assert args.handler(args) == 0

    assert capsys.readouterr().out.splitlines() == ["ID\tFLUX", "1\t5.5"]
