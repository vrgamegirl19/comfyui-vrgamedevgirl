"""Build a Typeform feature-report draft from the Beta2.0 test checklist.

The output contains no credentials. Submit it with a Typeform personal token held
outside the repository. Each response reports one feature, so all checklist cases
fit in the feature selector without forcing testers through 53 questions at once.
"""

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKLIST = ROOT / "docs" / "BETA2_TESTER_CHECKLIST.md"
CASE_RE = re.compile(r"^\*\*([A-Z]+-\d+) — (.+?)\.\*\* (.+)$")
CHECKLIST_URL = (
    "https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/"
    "pull/240/files"
)


def checklist_cases():
    cases = []
    for line in CHECKLIST.read_text(encoding="utf-8").splitlines():
        if match := CASE_RE.match(line):
            case_id, title, instruction = match.groups()
            cases.append((case_id, title, instruction))
    if not cases or len({case[0] for case in cases}) != len(cases):
        raise ValueError("Missing or duplicate Beta2.0 checklist cases")
    return cases


def field(ref, title, kind, *, required=False, description="", choices=None):
    result = {"ref": ref, "title": title, "type": kind,
              "validations": {"required": required}}
    properties = {}
    if description:
        properties["description"] = description
    if choices is not None:
        properties["choices"] = choices
    if properties:
        result["properties"] = properties
    return result


def jump(destination, condition):
    return {"action": "jump", "details": {"to": {"type": "field", "value": destination}},
            "condition": condition}


def result_is(choice):
    return {"op": "is", "vars": [
        {"type": "field", "value": "result"},
        {"type": "choice", "value": choice},
    ]}


def build_form():
    cases = checklist_cases()
    choices = [{"ref": case_id.lower().replace("-", "_"),
                "label": f"{case_id} — {title}"} for case_id, title, _ in cases]
    return {
        "title": "VRGDG Beta2.0 — feature test report",
        "type": "form",
        "settings": {"language": "en", "is_public": False, "show_progress_bar": True,
                     "show_question_number": True, "autosave_progress": True},
        "welcome_screens": [{
            "ref": "welcome", "title": "Help test Beta2.0",
            "properties": {"button_text": "Report a feature", "show_button": True,
                           "description": "Submit one report for each feature you try. The feature IDs and test steps are in the Beta2.0 checklist: "
                                          + CHECKLIST_URL + ". Do not paste API keys or private project paths."},
        }],
        "fields": [
            field("tester", "What name or handle should we use for your report?", "short_text", required=True),
            field("build", "What Beta2.0 build or commit are you testing?", "short_text",
                  description="Copy the build shown in the Builder header, or write unknown."),
            field("environment", "What is your setup?", "long_text",
                  description="OS, browser, ComfyUI version, GPU/VRAM, video engine and models relevant to this test."),
            field("feature", "Which Beta2.0 feature did you test?", "dropdown", required=True,
                  description="Select its ID from the checklist. Submit the form again for another feature.",
                  choices=choices),
            field("result", "Did this feature work?", "multiple_choice", required=True,
                  choices=[{"ref": "pass", "label": "Yes — passed"},
                           {"ref": "fail", "label": "No — failed"},
                           {"ref": "not_tested", "label": "I could not test it"}]),
            field("error", "What was the exact error message?", "long_text", required=True,
                  description="Copy and paste it exactly. If no message appeared, write 'No message'."),
            field("steps", "What happened, and how can we reproduce it?", "long_text", required=True,
                  description="Include expected versus actual behavior and the steps you took."),
            field("not_tested_reason", "What prevented you from testing it?", "long_text", required=True,
                  description="For example, a missing model, account, GPU, or project media."),
            field("notes", "Any other notes or a screenshot/log link?", "long_text",
                  description="Optional. Share only sanitized logs and public links."),
        ],
        "logic": [
            {"type": "field", "ref": "result", "actions": [
                jump("notes", result_is("pass")),
                jump("error", result_is("fail")),
                jump("not_tested_reason", result_is("not_tested")),
            ]},
            {"type": "field", "ref": "steps", "actions": [
                jump("notes", {"op": "always", "vars": []}),
            ]},
        ],
        "thankyou_screens": [{"ref": "thanks", "title": "Thank you for testing Beta2.0!",
                              "properties": {"show_button": False}}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="Path for the credential-free JSON draft")
    args = parser.parse_args()
    form = build_form()
    args.output.write_text(json.dumps(form, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(form['fields'])} questions and {len(form['fields'][3]['properties']['choices'])} feature choices")


if __name__ == "__main__":
    main()
