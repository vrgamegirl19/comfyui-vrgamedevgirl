"""Lossless prompt conversion and durable backups, without ComfyUI or paid LLM calls."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_seedance_conversion_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules.setdefault(PACKAGE, package)
conversion = importlib.import_module(f"{PACKAGE}.llm.seedance_conversion")

SOURCE = '''subject_definitions:
<Subject 1> is Wren from <Picture 1>, with silver-white braids, a black-rose antler crown and black lace gloves.

<Subject 2> is the violet-lit opera-house foyer from the second reference image.

<Audio 1> is the complete song, reused 1:1 as the final soundtrack.

retention_analysis:
<Subject 1>: fully_preserved - Keep her half-lace face paint and every silver teardrop jewel.

detailed_description:
[Shot 1] Low-angle tracking shot. <Subject 1> climbs the stairs, her lips closed.

[Shot 2] At 00:03.200, continuing directly from the previous shot's final moment, <Subject 1> sings "Never take my crown!" from 3.2 s to 5.8 s, then turns clockwise and snarls.

overall_soundscape:
<Audio 1> is reused without alteration. A thunderous gothic-metal rhythm.
'''


def scene():
    return {"id": "scene_1", "source_prompt": SOURCE, "minimax_h3_prompt": SOURCE,
            "image_references": [{"path": "wren.png"}, {"path": "foyer.png"}],
            "reference_aliases": {"<Subject 1>": "@Image 1", "<Subject 2>": "@Image 2"},
            "original_soundtrack": True}


def plan(blocks):
    return {"sections": [{"heading": block["heading_hint"], "block_ids": [block["id"]]} for block in blocks]}


class SeedanceConversionTests(unittest.TestCase):
    def test_every_immutable_block_survives_with_exact_words_cues_and_order(self):
        blocks = conversion.prepare_seedance_blocks(scene())
        result = conversion.assemble_seedance_prompt(blocks, plan(blocks))
        positions = [result.index(block["text"]) for block in blocks]
        self.assertEqual(positions, sorted(positions))
        for phrase in ['silver-white braids', 'half-lace face paint', 'every silver teardrop jewel',
                       '00:03.200', 'from 3.2 s to 5.8 s', '"Never take my crown!"',
                       'turns clockwise and snarls', 'previous shot\'s final moment', 'reused 1:1']:
            self.assertIn(phrase, result)
        self.assertIn('@Image 1 is Wren from @Image 1', result)
        self.assertIn('@Image 2 is the violet-lit', result)
        self.assertIn('Original soundtrack is the complete song', result)
        self.assertNotIn('<Subject', result)
        self.assertNotIn('<Picture', result)
        self.assertNotIn('fully_preserved', result)
        self.assertNotIn('@Audio 1', result)

    def test_missing_duplicate_or_reordered_blocks_are_rejected(self):
        blocks = conversion.prepare_seedance_blocks(scene())
        for ids in [[b['id'] for b in blocks][:-1], [b['id'] for b in blocks][::-1],
                    [b['id'] for b in blocks] + ['b1']]:
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, 'dropped, duplicated, or reordered'):
                conversion.assemble_seedance_prompt(blocks, {'sections': [{'heading': 'Shots', 'block_ids': ids}]})

    def test_llm_cannot_inject_rewritten_scene_text(self):
        blocks = conversion.prepare_seedance_blocks(scene())
        reply = plan(blocks)
        reply['sections'][0]['text'] = 'Wren is now wearing a green dress.'
        result = conversion.assemble_seedance_prompt(blocks, reply)
        self.assertNotIn('green dress', result)
        self.assertIn('black lace gloves', result)

    def test_reconverting_seedance_does_not_accumulate_section_headings(self):
        value = scene()
        blocks = conversion.prepare_seedance_blocks(value)
        first = conversion.assemble_seedance_prompt(blocks, plan(blocks))
        value['source_prompt'] = first
        second_blocks = conversion.prepare_seedance_blocks(value)
        second = conversion.assemble_seedance_prompt(second_blocks, plan(second_blocks))
        self.assertEqual(first, second)

    def test_audio_reference_is_only_used_when_actually_attached(self):
        value = scene()
        value['audio_paths'] = ['song_clip.wav']
        self.assertIn('@Audio 1', '\n'.join(b['text'] for b in conversion.prepare_seedance_blocks(value)))
        del value['audio_paths']
        value['original_soundtrack'] = False
        with self.assertRaisesRegex(ValueError, 'Cannot map <Audio 1>'):
            conversion.prepare_seedance_blocks(value)

    def test_minimax_delivery_and_emphasis_become_plain_seedance_text_without_losing_lyrics(self):
        value = scene()
        value['source_prompt'] = 'detailed_description:\n[Shot 1] From 0.2 s to 2.52 s, <Subject 1> sings in sync with the vocals in <Audio 1>, <d>[English, powerful, triumphant, singing] Take a <i>bow</i>, take a <i>bow</i></d>.'
        value['use_scene_audio'] = True
        blocks = conversion.prepare_seedance_blocks(value)
        result = conversion.assemble_seedance_prompt(blocks, plan(blocks))
        self.assertIn('"Take a bow, take a bow"', result)
        self.assertIn('(English, powerful, triumphant, singing)', result)
        self.assertIn('(emphasize "bow", "bow")', result)
        self.assertIn('0.2 s to 2.52 s', result)
        self.assertIn('vocals in @Audio 1', result)
        self.assertIn('@Audio 1 is the supplied project/custom audio excerpt', result)
        self.assertNotIn('<d>', result)
        self.assertNotIn('<i>', result)
        value['source_prompt'] = result
        again = conversion.prepare_seedance_blocks(value)
        self.assertEqual(conversion.assemble_seedance_prompt(again, plan(again)), result)

    def test_continuity_cleanup_preserves_the_action_and_only_replaces_the_stock_freeze_instruction(self):
        value = scene()
        value['source_prompt'] = "[Shot 1] All shots follow one continuous moment, preserving every person's exact position, posture, and movement across every camera cut; each shot continues from the final moment of the previous shot. Wren leaps, then raises her arm at 00:03.200."
        result = conversion.prepare_seedance_blocks(value)[0]['text']
        self.assertNotIn("preserving every person's exact position", result)
        self.assertIn('without resetting or freezing the action', result)
        self.assertIn('Wren leaps, then raises her arm at 00:03.200.', result)

    def test_staging_repairs_preserve_quoted_words_and_timing(self):
        value = scene()
        value['source_prompt'] = '''Shots:
[Shot 2] At 00:02.400, Hard cut, the same moment on the same stage, she sings "Hard cut, the same moment on the same stage" from 2.6 s to 4.06 s.

[Shot 3] At 00:04.100, a Dutch-tilted close-up of her spinning once as she sings "Don't look now (oh no!)" from 4.14 s to 6.16 s, then sings "spinning once".'''
        result = '\n'.join(b['text'] for b in conversion.prepare_seedance_blocks(value))
        self.assertIn('Hard cut, continuing immediately from the preceding shot on the same stage', result)
        self.assertIn('"Hard cut, the same moment on the same stage"', result)
        self.assertIn('making a partial turn toward the camera while keeping the singing mouth visible', result)
        for phrase in ['00:02.400', '2.6 s to 4.06 s', '00:04.100', '4.14 s to 6.16 s', '"Don\'t look now (oh no!)"', '"spinning once"']:
            self.assertIn(phrase, result)

    def test_turn_repair_is_scoped_to_the_vocal_closeup_shot(self):
        value = scene()
        value['source_prompt'] = '''Shots:
[Shot 1] A wide shot of her spinning once as she sings "Hello".
[Shot 2] A close-up of her spinning a full turn as she sings "Again".
[Shot 3] A close-up of a dancer spinning once, smiling.'''
        result = '\n'.join(b['text'] for b in conversion.prepare_seedance_blocks(value))
        self.assertIn('wide shot of her spinning once', result)
        self.assertIn('close-up of a dancer spinning once', result)
        self.assertEqual(result.count('making a partial turn'), 1)

    def test_vocal_guidance_preserves_parenthetical_lyrics_and_reconverts_once(self):
        value = scene()
        value['source_prompt'] = 'Shots:\n[Shot 1] The lead sings "Don\'t look now (oh no!)" from 0.38 s to 2.28 s.'
        value['use_scene_audio'] = True
        blocks = conversion.prepare_seedance_blocks(value)
        first = conversion.assemble_seedance_prompt(blocks, plan(blocks))
        self.assertIn('"Don\'t look now (oh no!)"', first)
        self.assertIn('Instrument-only performers keep their mouths closed', first)
        self.assertIn('Preserve the source\'s lead-vocal assignment for ambiguous phrases', first)
        self.assertNotIn('@Image 5', first)
        value['source_prompt'] = first
        again = conversion.prepare_seedance_blocks(value)
        self.assertEqual(conversion.assemble_seedance_prompt(again, plan(again)), first)
        value['use_scene_audio'] = False
        value['source_prompt'] = '[Shot 1] The lead sings "Hello".'
        self.assertNotIn(conversion.SEEDANCE_VOCAL_ROLE_GUIDANCE, '\n'.join(b['text'] for b in conversion.prepare_seedance_blocks(value)))

    def test_scene_audio_tags_are_not_invented_for_manual_reference_mode(self):
        value = scene()
        value['source_prompt'] = '[Shot 1] <Subject 1> sings in sync with <Audio 1>.'
        value['use_scene_audio'] = False
        blocks = conversion.prepare_seedance_blocks(value)
        self.assertIn('Original soundtrack', blocks[0]['text'])
        self.assertNotIn('@Audio 1', blocks[0]['text'])

    def test_invalid_or_missing_image_tags_fail_and_oversize_is_never_truncated(self):
        value = scene()
        value['reference_aliases']['<Subject 1>'] = '@Image 3'
        with self.assertRaisesRegex(ValueError, 'does not have an attached reference'):
            conversion.prepare_seedance_blocks(value)
        value['source_prompt'] = '<Picture 3> is the character.'
        with self.assertRaisesRegex(ValueError, 'Cannot map'):
            conversion.prepare_seedance_blocks(value)
        with self.assertRaisesRegex(ValueError, 'Nothing was truncated'):
            conversion.assemble_seedance_prompt([{'id': 'b1', 'text': 'x' * 15001}], {'sections': [{'heading': 'Shots', 'block_ids': ['b1']}]})

    def test_backups_are_unique_readable_and_exclude_runner_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            value = {**scene(), 'llm_api_key': 'never-back-up-this-key', 'api_video_prompt': 'previous Seedance prompt'}
            first = conversion.backup_seedance_prompts({'project_folder': folder, 'scenes': [value]})
            second = conversion.backup_seedance_prompts({'project_folder': folder, 'scenes': [value]})
            self.assertNotEqual(first['backup_path'], second['backup_path'])
            saved = json.loads(Path(first['backup_path']).read_text(encoding='utf-8'))
            self.assertEqual(saved['scenes'][0]['minimax_h3_prompt'], SOURCE)
            self.assertEqual(saved['scenes'][0]['api_video_prompt'], 'previous Seedance prompt')
            self.assertNotIn('never-back-up-this-key', json.dumps(saved))
            loaded = conversion.load_backup_scene({'project_folder': folder, 'backup_id': first['backup_id'], 'scene_id': 'scene_1'})
            self.assertEqual(loaded['source_prompt'], SOURCE)
            with self.assertRaises(ValueError):
                conversion.load_backup_scene({'project_folder': folder, 'backup_id': '../bad', 'scene_id': 'scene_1'})

    def test_conversion_reads_backed_up_source_and_uses_registered_instructions(self):
        with tempfile.TemporaryDirectory() as folder:
            backup = conversion.backup_seedance_prompts({'project_folder': folder, 'scenes': [scene()]})
            runner = Mock(return_value=(json.dumps(plan(conversion.prepare_seedance_blocks(scene()))), {'runner': 'test'}))
            instructions = Mock(side_effect=lambda payload, key, default: default)
            modules = {
                f'{PACKAGE}.llm.builder_runner': types.SimpleNamespace(_run_builder_text_llm=runner),
                f'{PACKAGE}.llm.builder_instructions': types.SimpleNamespace(_effective_builder_instruction=instructions),
                f'{PACKAGE}.llm.output_checks': types.SimpleNamespace(_extract_json_object_from_text=json.loads),
            }
            with patch.dict(sys.modules, modules):
                result = conversion.convert_seedance_prompt({'project_folder': folder, 'backup_id': backup['backup_id'], 'scene_id': 'scene_1', 'source_prompt': 'Do not use this unbacked text'})
            self.assertIn('Never take my crown!', result['prompt'])
            self.assertEqual(instructions.call_args.args[1], 'seedance25_prompt_conversion')
            self.assertIn('immutable', runner.call_args.args[1].lower())
            self.assertNotIn('Do not use this unbacked text', runner.call_args.args[1])
            self.assertEqual(Path(backup['backup_path']).read_text(encoding='utf-8').count('Never take my crown!'), 2)


if __name__ == '__main__':
    unittest.main()
