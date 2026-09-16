from pathlib import Path


UI_SOURCE = (Path(__file__).parents[1] / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")


def test_mapper_apply_keeps_stable_scene_link_for_instrumental_corrections():
    assert "lyric_mapper_line_id" in UI_SOURCE
    assert "If a mapper row was added while its corresponding scene is still" in UI_SOURCE
    assert "instrumental, preserve the user's line order as the final fallback." in UI_SOURCE
    assert "segment.lyric_text = String(line.text || \"\").trim();" in UI_SOURCE


def test_lyric_mapping_is_saved_in_both_directions():
    assert "function syncLyricMapperFromSegments()" in UI_SOURCE
    assert UI_SOURCE.count("syncLyricMapperFromSegments();") >= 3
    assert "applyLyricMapperToSegments({ overwriteSingers: true });" in UI_SOURCE


def test_unchecking_instrumental_restores_review_text():
    assert "text.dataset.reviewPreInstrumentalText" in UI_SOURCE
    assert "const restored = String(text.dataset.reviewPreInstrumentalText || \"\");" in UI_SOURCE


def test_instrumental_section_is_recomputed_after_lyric_text_is_added():
    assert 'existingSection !== "instrumental"' in UI_SOURCE
    assert 'if (!lyricText || isInstrumentalLyricText(lyricText))' in UI_SOURCE
    assert 'applyLyricSectionsFromReferenceText(state.segments, state.lyricMapper?.source_text || "");' in UI_SOURCE


def test_new_projects_and_new_reference_lyrics_clear_stale_mapper_lines():
    assert 'state.lyricMapper = defaultLyricMapper();' in UI_SOURCE
    assert 'normalizeLyricMapper(data.lyricMapper || data.lyric_mapper || {});' in UI_SOURCE
    assert 'source_text: String(options.referenceLyrics || "").trim(),\n        lines: [],' in UI_SOURCE
