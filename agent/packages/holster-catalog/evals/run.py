"""How often the interpret call routes cases.json's messages the way the prompt and
the tool schemas say it should.

Each case runs the interpret node alone against the live model: no TMDB request, no
rank or compose call. The score is printed, never asserted, because the model is not
deterministic and a miss is prompt work to look at, not a build to fail.

Run from agent/, where langchain and the Google provider are installed:

    uv run python packages/holster-catalog/evals/run.py           # live, GOOGLE_API_KEY
    uv run python packages/holster-catalog/evals/run.py --check   # validate cases.json
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, get_args

from langchain.chat_models import init_chat_model
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.runtime import Runtime
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from holster_catalog.catalog_tool import CATALOG_TOOLS, CatalogToolError
from holster_catalog.graph import TurnContext, interpret_node
from holster_catalog.testing import FakeTMDB
from holster_catalog.tmdb import MediaType, TMDBClient, is_genre_resolved
from holster_catalog.turn import Turn

CASES = Path(__file__).with_name("cases.json")

# Mirrors agent/main.py's DEFAULT_MODEL and the model it builds in lifespan.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Under the free tier's 15 requests a minute, with room for an occasional retry.
PACE_SECONDS = 4.5

PROVIDERS = ["Netflix", "Hulu"]


def _tool_name(schema: type[BaseModel]) -> str:
    """The name the model is offered the schema under."""
    name: str = convert_to_openai_tool(schema)["function"]["name"]
    return name


TOOLS = {_tool_name(t): t for t in CATALOG_TOOLS}

# Prose the model writes afresh every call; no expected value can match it.
FREE_TEXT = frozenset({"reply", "text"})


class Case(BaseModel):
    # forbid, so a misspelt key fails --check instead of dropping the context
    # the case is about.
    model_config = ConfigDict(extra="forbid")

    id: str
    message: str
    history: list[Turn] = Field(default_factory=list)
    tool: str
    # A subset of the tool's fields: only the ones the case is about.
    args: dict[str, Any]

    @model_validator(mode="after")
    def _args_fit_the_tool(self) -> Case:
        schema = TOOLS.get(self.tool)
        if schema is None:
            raise ValueError(f"tool {self.tool!r} is not one of {sorted(TOOLS)}")
        for key, value in self.args.items():
            field = schema.model_fields.get(key)
            if field is None or key in FREE_TEXT:
                raise ValueError(
                    f"{self.id}: {key!r} is not a comparable field of {self.tool}"
                )
            # One field through the schema's own validators, on an instance built
            # without the fields this case leaves out, which is sound while no
            # validator reads another field (a model validator, or info.data).
            # Strict, and compared after: a value they would reject, coerce, clamp
            # or rewrite can never equal what the model's parsed reply holds.
            probe = schema.model_construct()
            try:
                schema.__pydantic_validator__.validate_assignment(
                    probe, key, value, strict=True
                )
            except ValidationError as exc:
                raise ValueError(f"{self.id}: {key}: {exc.errors()[0]['msg']}") from exc
            if getattr(probe, key) != value:
                raise ValueError(
                    f"{self.id}: {key}: the schema turns {value!r} "
                    f"into {getattr(probe, key)!r}"
                )
        # A case may leave media_type out, and then either answer is the model's
        # to give, so its genres must be real genres for both.
        media_types: tuple[MediaType, ...] = (
            (self.args["media_type"],)
            if "media_type" in self.args
            else get_args(MediaType)
        )
        for key in ("genres", "without_genres"):
            for name in self.args.get(key, []):
                if not all(is_genre_resolved(m, name) for m in media_types):
                    raise ValueError(
                        f"{self.id}: {key}: {name!r} is not a genre for {media_types}"
                    )
        return self


def load_cases() -> list[Case]:
    cases = TypeAdapter(list[Case]).validate_json(CASES.read_bytes())
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids are not unique")
    if not cases:
        raise ValueError("cases.json holds no cases")
    return cases


def _comparable(value: Any) -> Any:
    """Case, edge whitespace and order mean nothing where these values land:
    holster_catalog looks genres and names up by their stripped, lower-cased form,
    and every list field is a set of filters. It matches titles more loosely still,
    ignoring accents and punctuation, so a title differing only in those is a miss
    here that the lookup would find."""
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, list):
        return frozenset(_comparable(v) for v in value)
    return value


def _mismatches(case: Case, picked: BaseModel) -> list[str]:
    picked_tool = _tool_name(type(picked))
    if picked_tool != case.tool:
        return [f"tool {picked_tool}, wanted {case.tool}"]
    return [
        f"{key}={getattr(picked, key)!r}, wanted {want!r}"
        for key, want in case.args.items()
        if _comparable(getattr(picked, key)) != _comparable(want)
    ]


def _is_bad_reply(exc: CatalogToolError) -> bool:
    """True when the model replied and the reply is the prompt's to fix: no tool or
    several, which interpret raises itself with no cause, or an undeclared tool or
    arguments its schema rejects, the parser's OutputParserException or
    ValidationError that _model_call wraps. A failed request, such as a rate
    limit, a 404 or a timeout, arrives as the provider's or the transport's own
    error."""
    return exc.__cause__ is None or isinstance(
        exc.__cause__, (OutputParserException, ValidationError)
    )


async def _run(cases: list[Case], model: BaseChatModel, client: TMDBClient) -> None:
    interpret = interpret_node(model)
    runtime = Runtime(
        context=TurnContext(
            client=client,
            watch_region="US",
            watch_providers=[8],
            watch_provider_names=PROVIDERS,
            verdicts=[],
            shown=[],
        )
    )
    loop = asyncio.get_running_loop()
    next_start = loop.time()
    agreed = errors = 0
    for case in cases:
        # Start to start, so the call's own time counts toward the pace.
        await asyncio.sleep(max(0.0, next_start - loop.time()))
        next_start = loop.time() + PACE_SECONDS
        try:
            state = await interpret(
                {"message": case.message, "history": case.history}, runtime
            )
        # A failed request, such as a rate limit or a timeout, is left out of the
        # score. A reply that picks no tool, several, an undeclared one, or arguments
        # its schema rejects is the prompt's to fix, so it counts as a miss.
        except CatalogToolError as exc:
            if _is_bad_reply(exc):
                print(f"MISS  {case.id}: {exc}")
            else:
                errors += 1
                print(f"ERROR {case.id}: {exc}")
            continue
        wrong = _mismatches(case, state["intent"])
        if wrong:
            print(f"MISS  {case.id}: {'; '.join(wrong)}")
        else:
            agreed += 1
            print(f"ok    {case.id}")
    ran = len(cases) - errors
    share = f"{agreed / ran:.0%}" if ran else "-"
    print(f"\nagreement {agreed}/{ran} ({share}), {errors} errors")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the interpret call on cases.json."
    )
    parser.add_argument(
        "--check", action="store_true", help="validate cases.json and call nothing"
    )
    args = parser.parse_args()
    cases = load_cases()
    if args.check:
        print(f"{len(cases)} cases valid")
        return
    if not os.environ.get("GOOGLE_API_KEY", "").strip():
        sys.exit("GOOGLE_API_KEY not set")
    model_name = os.getenv("GOOGLE_MODEL") or DEFAULT_MODEL
    model = init_chat_model(f"google_genai:{model_name}", timeout=15.0, max_retries=2)
    print(f"model {model_name}, {len(cases)} cases\n")

    async def run() -> None:
        # TurnContext requires a client, and the interpret node never calls it.
        async with FakeTMDB().client() as client:
            await _run(cases, model, client)

    asyncio.run(run())


if __name__ == "__main__":
    main()
