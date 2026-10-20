"""Keeps LangSmith tracing off for the suite whatever the shell exports.

Traced, a fake turn's runs are uploaded, and serializing a run's inputs for
the upload drains the scripted model's replies (testutil.scripted) before the
model reads them, so the turn fails. `langsmith.configure`'s setting is
checked before the environment variables."""

import langsmith

langsmith.configure(enabled=False)
