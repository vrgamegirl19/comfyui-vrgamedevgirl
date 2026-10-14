import ast
import json
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / 'llm/api.py').read_text(encoding='utf-8'))
multi = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'VRGDG_LLM_Multi')
TABLES = {
    node.targets[0].id: ast.literal_eval(node.value)
    for node in multi.body if isinstance(node, ast.Assign)
    and node.targets[0].id in {'PROVIDER_MODELS', 'DEFAULT_MODEL'}
}


class ModelCatalogTests(unittest.TestCase):
    def test_model_lists_are_unique_and_defaults_remain_selectable(self):
        for provider, models in TABLES['PROVIDER_MODELS'].items():
            with self.subTest(provider=provider):
                self.assertEqual(len(models), len(set(models)))
                self.assertIn(TABLES['DEFAULT_MODEL'][provider], models)
                self.assertTrue(all(model.strip() == model and model for model in models))

    def test_api_choices_expose_the_complete_catalog_to_builder_runners(self):
        tree = ast.parse((ROOT / 'llm/builder_runner.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_llm_multi_choices')
        api = types.ModuleType('catalog_test.api')
        api.VRGDG_LLM_Multi = type('Multi', (), TABLES)
        namespace = {'__package__': 'catalog_test'}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'choices', 'exec'), namespace)
        with patch.dict(sys.modules, {'catalog_test.api': api}):
            choices = namespace['_llm_multi_choices']()['providers']
        self.assertEqual({row['id']: row['models'] for row in choices}, TABLES['PROVIDER_MODELS'])
        self.assertEqual({row['id']: row['default_model'] for row in choices}, TABLES['DEFAULT_MODEL'])

    def test_frontend_catalog_matches_backend_and_preserves_selected_models(self):
        script = r'''
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8').replace(/^import .*;\r?\n/m, '');
const context = vm.createContext({ app: { registerExtension() {}, graph: { setDirtyCanvas() {} } }, setTimeout() {} });
vm.runInContext(source, context);
const tables = vm.runInContext('({PROVIDER_MODELS, DEFAULT_MODEL})', context);
for (const [provider, models] of Object.entries(tables.PROVIDER_MODELS)) {
  for (const selected of [models[0], tables.DEFAULT_MODEL[provider]]) {
    const widget = { name: 'model', value: selected };
    const node = { widgets: [{name: 'provider', value: provider}, widget], size: [400, 200], computeSize: () => [400, 200], setSize() {} };
    context.syncModelWidget(node);
    if (widget.value !== selected || JSON.stringify(widget.options.values) !== JSON.stringify(models)) throw Error(provider);
  }
}
process.stdout.write(JSON.stringify(tables));
'''
        result = subprocess.run(['node', '-e', script, str(ROOT / 'web/VRGDG_LLM_Multi_dynamic.js')], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout), TABLES)

    def test_current_provider_models_are_available(self):
        required = {
            'openai': ['gpt-6.1-sol'],
            'anthropic': ['claude-fable-5-1', 'claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5'],
            'google': ['gemini-4-argon', 'gemini-3.8-flash', 'gemini-3.6-flash', 'gemini-3.5-flash-lite'],
            'grok': ['grok-4.7'],
            'deepseek': ['deepseek-flash', 'deepseek-v4-pro'],
            'openrouter': ['openai/gpt-6.1-sol', 'google/gemini-4-argon', 'anthropic/claude-haiku-5.5', 'x-ai/grok-4.7', 'deepseek/deepseek-v4.1-flash'],
        }
        for provider, models in required.items():
            with self.subTest(provider=provider):
                self.assertTrue(set(models).issubset(TABLES['PROVIDER_MODELS'][provider]))
        self.assertFalse(any(':batch' in model for model in TABLES['PROVIDER_MODELS']['openrouter']))


if __name__ == '__main__':
    unittest.main()
