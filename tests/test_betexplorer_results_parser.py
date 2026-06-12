from __future__ import annotations

from pathlib import Path

from scripts.parse_betexplorer_results_pages import parse_page


def test_parse_page_reads_nested_data_odd_attributes(tmp_path: Path) -> None:
    page = tmp_path / "results_2014_betexplorer_main.html"
    page.write_text(
        """
        <table><tr>
          <td><a><span>Brazil</span> - <span>Croatia</span></a></td>
          <td>3:1</td>
          <td data-odd="1.35"></td>
          <td><span data-odd="5.20"></span></td>
          <td data-odd="9.10"></td>
          <td>12.06.2014</td>
        </tr></table>
        """,
        encoding="utf-8",
    )
    parsed = parse_page(page, 2014, "group_stage")

    assert len(parsed) == 1
    assert parsed.loc[0, "team_a"] == "Brazil"
    assert parsed.loc[0, "team_b"] == "Croatia"
    assert parsed.loc[0, ["odds_team_a", "odds_draw", "odds_team_b"]].tolist() == [1.35, 5.20, 9.10]
