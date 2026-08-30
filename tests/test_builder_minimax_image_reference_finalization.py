from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")


def _function_source(start_marker: str, end_marker: str) -> str:
    start = BUILDER_SOURCE.index(start_marker)
    end = BUILDER_SOURCE.index(end_marker, start)
    return BUILDER_SOURCE[start:end]


def test_image_reference_authority_stays_out_of_render_finalization():
    prompt_context = _function_source(
        "  function miniMaxH3PromptContextForSegment(segment, mode) {",
        "  function miniMaxH3PromptVisionImages(segment, mode) {",
    )
    render_all = _function_source(
        "  async function renderAllScenes(options = {}) {",
        "  async function zImageAllScenes(options = {}) {",
    )

    assert "IMAGE + REFERENCE FRAME AUTHORITY — MANDATORY:" in prompt_context
    assert "IMAGE + REFERENCE FRAME AUTHORITY — MANDATORY:" not in render_all
    assert 'if (mode === "image_reference_to_video")' not in render_all
