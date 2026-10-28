import logging

import pandas as pd
import pytest

import data


@pytest.fixture
def characters():
    # Small stand-in for the Marvel csv so tests don't depend on the real file
    return pd.DataFrame(
        {
            "name": [
                "Spider-Man (Peter Parker)",
                "Captain America (Steven Rogers)",
                None,
                "Peter Parker (Kaine) (Earth-616)",
                "Thor (Thor Odinson)",
                "Iron Man (Anthony Stark)",
            ]
        }
    )


def test_add_bodycount_marks_only_peter_parker(characters):
    result = data.add_bodycount(characters)
    assert result["bodycount"].tolist() == [3, 0, 0, 3, 0, 0]


def test_add_bodycount_leaves_input_untouched(characters):
    data.add_bodycount(characters)
    assert "bodycount" not in characters.columns


def test_add_bodycount_rejects_missing_name_column():
    with pytest.raises(data.MissingColumnsError, match="'Name' looks like 'name'"):
        data.add_bodycount(pd.DataFrame({"Name": ["Spider-Man (Peter Parker)"]}))


def test_bodycount_preview_returns_slices(characters):
    peter_parker, head = data.bodycount_preview(data.add_bodycount(characters), n=2)

    assert peter_parker.index.tolist() == [0, 3]
    assert peter_parker["bodycount"].tolist() == [3, 3]
    assert list(head.columns) == ["name", "bodycount"]
    assert head["bodycount"].tolist() == [3, 0]


def test_main_logs_preview_at_info(tmp_path, characters, caplog):
    csv = tmp_path / "marvel.csv"
    characters.to_csv(csv, index=False)

    with caplog.at_level(logging.INFO, logger=data.__name__):
        data.main(["--csv", str(csv)])

    assert "Peter Parker's Row" in caplog.text
    assert "First 5 Rows" in caplog.text


def test_main_quiet_at_warning(tmp_path, characters, caplog, capsys):
    csv = tmp_path / "marvel.csv"
    characters.to_csv(csv, index=False)

    with caplog.at_level(logging.WARNING, logger=data.__name__):
        data.main(["--csv", str(csv), "--log-level", "WARNING"])

    assert caplog.records == []
    assert capsys.readouterr().out == ""


def test_main_exits_on_missing_csv(tmp_path):
    with pytest.raises(SystemExit, match="Marvel csv not found"):
        data.main(["--csv", str(tmp_path / "nope.csv")])
