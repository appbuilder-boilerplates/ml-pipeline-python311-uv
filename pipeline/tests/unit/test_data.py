from pathlib import Path

import pytest

from ml_pipeline import load_data, read_table


def test_load_data_keeps_labels_as_the_file_holds_them_and_ids_as_text(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("id,size,colour,label\n7,1.5,red,yes\n8,2.0,blue,no\n")
    data = load_data(path)
    assert data["ids"] == ["7", "8"]
    assert data["labels"] == ["yes", "no"]
    assert list(data["features"].columns) == ["size", "colour"]


def test_load_data_uses_the_named_columns(tmp_path: Path) -> None:
    path = tmp_path / "rows.tsv"
    path.write_text("row\tx\ttarget\na\t1\t0.5\nb\t2\t1.5\n")
    data = load_data(path, label_column="target", id_column="row")
    assert data["ids"] == ["a", "b"]
    assert data["labels"] == [0.5, 1.5]
    assert list(data["features"].columns) == ["x"]


def test_load_data_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": 1, "x": 3, "label": "a"}\n{"id": 2, "x": 4, "label": "b"}\n')
    assert load_data(path)["labels"] == ["a", "b"]


def test_load_data_refuses_a_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("id,x\n1,2\n")
    with pytest.raises(ValueError, match="missing column"):
        load_data(path)


def test_load_data_refuses_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("id,x,label\n1,2,a\n1,3,b\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_data(path)


def test_read_table_refuses_an_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        read_table(tmp_path / "rows.parquet")
