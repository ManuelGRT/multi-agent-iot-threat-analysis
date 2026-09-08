from __future__ import annotations

import pytest

from src.eval import reporting


def test_export_report_confines_output(tmp_path):
    path = reporting.export_report(
        "Informe",
        [{"heading": "Resultado", "body": {"ok": True}}],
        filename="reports/result.md",
        output_root=tmp_path,
    )

    assert path == (tmp_path / "reports" / "result.md").resolve()
    assert '"ok": true' in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("filename", ["../outside.md", "folder/../../outside.md"])
def test_export_report_rejects_traversal(tmp_path, filename):
    with pytest.raises(ValueError):
        reporting.export_report("Informe", [], filename=filename, output_root=tmp_path)
