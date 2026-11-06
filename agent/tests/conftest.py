"""Keeps LangSmith tracing off whatever the shell exports: uploading a fake turn's
run drains the scripted model's replies, and `langsmith.configure` outranks the
environment."""

import langsmith

langsmith.configure(enabled=False)
