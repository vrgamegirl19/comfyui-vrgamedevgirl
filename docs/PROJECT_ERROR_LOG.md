# Project error log

Open **Menu → View Error Log** in Video Builder to read the current project's
`logs/error_log.txt`. It is created when a saved project is opened or saved.
Errors append to this file; previous entries remain until **Clear Log** is
confirmed. **Refresh** reads new entries and **Copy Log** copies the displayed report.

Entries include a local timestamp with timezone offset, severity, original message,
and available scene, mode, stage, source, component, error details and stack trace.
Sources include frontend error toasts, Builder browser exceptions, project backend/API and background-job failures, and failed
ComfyUI prompts polled by the Builder. Errors handled by an endpoint without a
traceback retain its returned message; the logger does not fabricate a traceback.
The same API failure and its subsequent error toast produce one file entry.

Logs stay local. Known project, runner and request credentials, bearer tokens and
credential fields are redacted. Request bodies and model inputs are not recorded.
Failure to write the log does not stop the workflow or produce recursive toasts.
Before a project has been saved, errors still appear in the browser console.

Local Builder endpoints (POST, JSON body):

- `/vrgdg/music_builder/error_log`: `{ "project_folder": "..." }`; returns `path`, `text`.
- `/vrgdg/music_builder/error_log/append`: `{ "project_folder": "...", "entry": { "message": "..." } }`.
  Optional entry fields: `severity` (`ERROR`, `WARNING`, `CRITICAL`), `scene`, `stage`,
  `mode`, `source`, `component`, `details`, `stack`.
- `/vrgdg/music_builder/error_log/clear`: `{ "project_folder": "..." }`; retains the header.

Each endpoint requires an existing project session and uses the fixed contained log
path. Arbitrary log filenames are not accepted.
