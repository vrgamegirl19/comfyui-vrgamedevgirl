"""Contract tests for the Agent API v1 OpenAPI snapshot (Apireport K6, D12, Section 11 item 4).

``agent_api/openapi.json`` is the contract. These tests fail when the router, the
schemas or the error codes change without the snapshot being regenerated, so a
contract change is always a deliberate, reviewed diff.
"""

import importlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

_spec = importlib.util.spec_from_file_location("vrgdg_export_openapi", ROOT / "scripts" / "export_openapi.py")
export_openapi = importlib.util.module_from_spec(_spec)
sys.modules["vrgdg_export_openapi"] = export_openapi
_spec.loader.exec_module(export_openapi)

pkg_name = ROOT.name
router = importlib.import_module(f"{pkg_name}.agent_api.router")
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
envelope = importlib.import_module(f"{pkg_name}.agent_api.envelope")
schemas = importlib.import_module(f"{pkg_name}.agent_api.schemas")

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
_OPTIONAL_MINIMAX_KEYS = {
    f"pass{n}_use_{name}" for n in (1, 2) for name in ("te_speed", "feedforward", "block_sparse_attention")
} | {"ref_pass_profiles"}


def _snapshot():
    return json.loads(export_openapi.OPENAPI_PATH.read_text(encoding="utf-8"))


def _snapshot_operations():
    document = _snapshot()
    return {
        (method, path)
        for path, item in document["paths"].items()
        for method in item
        if method in HTTP_METHODS
    }


def _registered_routes():
    """Run the real route registration against a recording server and return (method, path)."""
    recorded = set()
    server = MagicMock()
    server.routes = MagicMock()
    for method in HTTP_METHODS:
        def make(method_name):
            def decorator_factory(path, *args, **kwargs):
                recorded.add((method_name, path))
                return lambda handler: handler
            return decorator_factory
        setattr(server.routes, method, MagicMock(side_effect=make(method)))

    previous = router._VRGDG_AGENT_API_ROUTES_REGISTERED
    router._VRGDG_AGENT_API_ROUTES_REGISTERED = False
    try:
        router.register_agent_api_routes(server)
    finally:
        router._VRGDG_AGENT_API_ROUTES_REGISTERED = previous
    return recorded


class AgentApiContractTests(unittest.TestCase):
    def test_snapshot_matches_router_and_schemas(self):
        expected = export_openapi.render(export_openapi.build_openapi())
        actual = export_openapi.OPENAPI_PATH.read_text(encoding="utf-8")
        self.assertEqual(
            actual,
            expected,
            "agent_api/openapi.json is out of date. Run scripts/export_openapi.py and "
            "review the diff; a contract change needs a version decision.",
        )

    def test_every_registered_route_is_in_the_contract_and_back(self):
        registered = _registered_routes()
        documented = _snapshot_operations()
        self.assertTrue(registered, "route registration recorded no routes")
        self.assertEqual(
            sorted(registered - documented),
            [],
            "Routes registered by router.py but missing from openapi.json.",
        )
        self.assertEqual(
            sorted(documented - registered),
            [],
            "Routes in openapi.json that router.py does not register.",
        )

    def test_operations_are_well_formed(self):
        document = _snapshot()
        self.assertTrue(document["openapi"].startswith("3.1"))
        operation_ids = []
        for path, item in document["paths"].items():
            self.assertTrue(path.startswith("/vrgdg/api/v1/"), path)
            braces = set(export_openapi._PATH_PARAM.findall(path))
            for method, operation in item.items():
                label = f"{method.upper()} {path}"
                operation_ids.append(operation["operationId"])
                path_params = {p["name"] for p in operation["parameters"] if p["in"] == "path"}
                self.assertEqual(path_params, braces, f"{label}: path parameters differ from the URL")
                self.assertIn("default", operation["responses"], f"{label}: no error response")
                success = [code for code in operation["responses"] if code != "default"]
                self.assertEqual(len(success), 1, f"{label}: expected exactly one success status")
                self.assertTrue(200 <= int(success[0]) < 300, f"{label}: success status {success[0]}")
                # DELETE may carry a body (OpenAPI 3.1 allows it); GET must not.
                if method == "get":
                    self.assertNotIn("requestBody", operation, f"{label}: unexpected request body")
        self.assertEqual(len(operation_ids), len(set(operation_ids)), "operationId values must be unique")

    def test_error_codes_match_errors_module(self):
        module_codes = sorted(
            value for name, value in vars(errors).items()
            if name.isupper() and isinstance(value, str) and value == name
        )
        self.assertEqual(_snapshot()["components"]["schemas"]["ErrorCode"]["enum"], module_codes)

    def test_envelopes_match_component_schemas(self):
        components = _snapshot()["components"]["schemas"]
        success = json.loads(envelope.api_success(data={"x": 1}, revision=3).body)
        failure = json.loads(envelope.api_error(code=errors.VALIDATION_ERROR, message="bad", status=400).body)
        for key in components["SuccessEnvelope"]["required"]:
            self.assertIn(key, success)
        self.assertEqual(set(success), set(components["SuccessEnvelope"]["properties"]))
        for key in components["ErrorEnvelope"]["required"]:
            self.assertIn(key, failure)
        self.assertEqual(set(failure["error"]), set(components["ErrorBody"]["properties"]))
        self.assertIn(failure["error"]["code"], components["ErrorCode"]["enum"])

    def test_effective_settings_match_component_schema(self):
        components = _snapshot()["components"]["schemas"]
        effective = schemas.extract_effective_settings({})
        self.assertEqual(set(effective), set(components["EffectiveSettings"]["properties"]))
        for group, ref in (
            (name, prop.get("$ref")) for name, prop in components["EffectiveSettings"]["properties"].items()
        ):
            if ref:
                schema_name = ref.rsplit("/", 1)[-1]
                documented = set(components[schema_name]["properties"])
                if group == "minimax_h3":
                    # Optional keys (per-pass toggles, profile cache) appear only once a project saves them.
                    self.assertTrue(set(effective[group]) <= documented, f"{set(effective[group]) - documented} undocumented")
                    self.assertTrue(documented - set(effective[group]) <= _OPTIONAL_MINIMAX_KEYS)
                    continue
                self.assertEqual(
                    set(effective[group]),
                    documented,
                    f"settings group '{group}' differs from schema {schema_name}",
                )

    def test_contract_version_is_semver(self):
        parts = _snapshot()["info"]["version"].split(".")
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(part.isdigit() for part in parts))


if __name__ == "__main__":
    unittest.main()
