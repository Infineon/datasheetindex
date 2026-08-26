"""Pydantic models for extraction input/output.

Note: Numerical fields use a sentinel value (NOT_SPECIFIED = -999.0) instead of
Optional[float] because the Claude Agent SDK's StructuredOutput validator does not
reliably handle anyOf/null JSON schema patterns. Use ConditionedValue.has_min()/
has_max()/has_typical() helpers to check for presence.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, Literal, NotRequired, TypedDict, cast

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

# Sentinel for "value not specified in the datasheet".
# Using Optional[float] would produce anyOf in the JSON schema, which the
# SDK's StructuredOutput validator cannot handle reliably.
NOT_SPECIFIED: Final[float] = -999.0


# The extraction output formats, named once so call sites that build a format
# from a lookup (compare._KIND_TO_FORMAT) can be typed instead of widening to str.
OutputFormat = Literal["auto", "numerical", "boolean", "list", "text"]


class ParameterSpec(BaseModel):
    """Defines a parameter to extract."""

    name: str
    output_format: OutputFormat = "auto"
    description: str = ""


class BoxPct(BaseModel):
    """A bounding box in page-normalized coordinates (0.0-1.0)."""

    model_config = {"extra": "forbid"}

    top: float = Field(ge=0.0, le=1.0)
    bottom: float = Field(ge=0.0, le=1.0)
    left: float = Field(ge=0.0, le=1.0)
    right: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> BoxPct:
        if self.left > self.right:
            raise ValueError("left must be <= right")
        if self.top > self.bottom:
            raise ValueError("top must be <= bottom")
        return self


class BoxPoints(BaseModel):
    """A bounding box in raw PDF points."""

    model_config = {"extra": "forbid"}

    x0: float
    y0: float
    x1: float
    y1: float  # PDF points (origin-relative; may be negative on edge glyphs)

    @model_validator(mode="after")
    def _ordered(self) -> BoxPoints:
        if self.x0 > self.x1:
            raise ValueError("x0 must be <= x1")
        if self.y0 > self.y1:
            raise ValueError("y0 must be <= y1")
        return self


class BoxPctWire(TypedDict):
    """Serialized shape of BoxPct (normalized coordinates 0.0-1.0)."""

    top: float
    bottom: float
    left: float
    right: float


class BoxPointsWire(TypedDict):
    """Serialized shape of BoxPoints (raw PDF points)."""

    x0: float
    y0: float
    x1: float
    y1: float


class SourceLocationWire(TypedDict):
    """Serialized shape of SourceLocation.

    A MIRROR of the model, not the model itself. The serializers below emit
    `source_location.model_dump()` -- a plain dict, recursively -- and
    annotating the field with the SourceLocation class instead makes Pydantic
    emit PydanticSerializationUnexpectedValue on every dump. Verified
    2026-08-19.
    """

    page: int
    match_method: Literal["search_for", "tokens"]
    page_width: float
    page_height: float
    region_pct: BoxPctWire
    region_points: BoxPointsWire


class ConditionedValueWire(TypedDict):
    """Serialized shape of ConditionedValue.

    Every key is NotRequired because _serialize omits empty strings and sentinel
    numerics. This annotation is the ONLY reason the published OpenAPI document
    has a real schema here rather than a bare object -- keep it in step with
    _serialize (pinned by tests/test_models_wire_schema.py).
    """

    conditions: NotRequired[str]
    typical_value: NotRequired[float]
    min_value: NotRequired[float]
    max_value: NotRequired[float]
    unit: NotRequired[str]
    source_text: NotRequired[str]
    source_location: NotRequired[SourceLocationWire]


class ParameterResultWire(TypedDict):
    """Serialized shape of ParameterResult. Only bool_value is optional -- it is
    omitted when the tri-state is "not_specified", so output JSON never carries a
    boolean the datasheet did not state."""

    parameter: str
    found: bool
    confidence: float
    reason: str
    page_numbers: list[int]
    original_terminology: str
    values: list[ConditionedValueWire]
    list_value: list[str]
    text_value: str
    source_text: str
    source_location: SourceLocationWire | None
    bool_value: NotRequired[bool]


class SourceLocation(BaseModel):
    """Engine-authored: where a value's source_text sits on the page.

    Populated by the source-grounding post-pass (see source_locator.py), never
    from model output. Once wired in, it is stripped from the model-facing
    schema via ENGINE_AUTHORED_FIELDS (added in the engine-authored-stripping
    task), exactly like process_diagnostics. Within-page sanity is enforced via
    the normalized region_pct bounds (0.0-1.0), not via absolute points, because
    raw points are page-origin-relative and locate_text does not clamp them.
    """

    model_config = {"extra": "forbid"}

    page: int = Field(ge=1)  # 1-indexed
    match_method: Literal["search_for", "tokens"]
    page_width: float = Field(gt=0.0)  # PDF points
    page_height: float = Field(gt=0.0)
    region_pct: BoxPct
    region_points: BoxPoints


class ConditionedValue(BaseModel):
    """A single set of values under specific test conditions."""

    # `extra=forbid` so JSON-schema generation emits ``additionalProperties:
    # false`` on this nested object. Required by OpenAI / Azure structured-
    # output strict mode (Anthropic's validator is permissive about it). Its
    # siblings ParameterResult / ClaimResult already declare this; ConditionedValue
    # was the only nested type in the chamber path that didn't.
    model_config = {"extra": "forbid"}

    conditions: str = ""
    min_value: float = Field(default=NOT_SPECIFIED)
    max_value: float = Field(default=NOT_SPECIFIED)
    typical_value: float = Field(default=NOT_SPECIFIED)
    unit: str = ""
    source_text: str = ""
    source_location: SourceLocation | None = None

    def has_min(self) -> bool:
        return self.min_value != NOT_SPECIFIED

    def has_max(self) -> bool:
        return self.max_value != NOT_SPECIFIED

    def has_typical(self) -> bool:
        return self.typical_value != NOT_SPECIFIED

    @model_serializer
    def _serialize(self) -> ConditionedValueWire:
        """Exclude sentinel values so output JSON only contains real data."""
        data: dict = {}
        if self.conditions:
            data["conditions"] = self.conditions
        if self.has_typical():
            data["typical_value"] = self.typical_value
        if self.has_min():
            data["min_value"] = self.min_value
        if self.has_max():
            data["max_value"] = self.max_value
        if self.unit:
            data["unit"] = self.unit
        if self.source_text:
            data["source_text"] = self.source_text
        if self.source_location is not None:
            # model_dump() (python mode) is fine: every SourceLocation field is a
            # primitive, so python/json modes are identical. Revisit if a non-
            # primitive field is ever added.
            data["source_location"] = self.source_location.model_dump()
        return cast(ConditionedValueWire, data)


class ParameterResult(BaseModel):
    """Extraction result for a single parameter."""

    model_config = {"extra": "forbid"}

    parameter: str
    found: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    page_numbers: list[int] = Field(default_factory=list)
    original_terminology: str = ""
    values: list[ConditionedValue] = Field(default_factory=list)
    # Non-numerical value types
    # bool_value is a TRI-STATE enum, not a plain bool: "true"/"false" record a
    # determined yes/no, and "not_specified" (the default) records "the datasheet
    # does not say / the model could not determine it". A plain bool cannot
    # represent that third state, so an unset boolean used to default to False and
    # render a spurious "no" (and lose comparison verdicts). The enum is a clean
    # string -- no anyOf/null -- so the zero-anyOf schema invariant holds, exactly
    # like the numeric NOT_SPECIFIED sentinel. Use has_bool()/bool_or_none() to read.
    bool_value: Literal["true", "false", "not_specified"] = "not_specified"
    list_value: list[str] = Field(default_factory=list)
    text_value: str = ""
    source_text: str = ""
    source_location: SourceLocation | None = None

    @field_validator("bool_value", mode="before")
    @classmethod
    def _coerce_bool_value(cls, v: Any) -> Any:
        """Coerce a Python bool/None into the tri-state enum.

        Lets legacy callers and serialized output (which emits a real JSON bool;
        see _serialize) round-trip back through model_validate, and keeps existing
        ``ParameterResult(bool_value=True)`` call sites working. Valid enum
        strings pass through; anything else falls through to enum validation.
        """
        if isinstance(v, bool):
            return "true" if v else "false"
        if v is None:
            return "not_specified"
        return v

    def has_bool(self) -> bool:
        """True when a yes/no was determined (i.e. not 'not_specified')."""
        return self.bool_value != "not_specified"

    def bool_or_none(self) -> bool | None:
        """The determined boolean, or None when not specified."""
        if self.bool_value == "true":
            return True
        if self.bool_value == "false":
            return False
        return None

    @model_serializer
    def _serialize(self) -> ParameterResultWire:
        """Serialize for output. Mirrors default field output, except bool_value
        is emitted as a real JSON bool when determined and OMITTED when
        not_specified -- so output JSON only carries a boolean that the datasheet
        actually stated (the sibling of ConditionedValue's sentinel stripping).
        """
        data: dict = {
            "parameter": self.parameter,
            "found": self.found,
            "confidence": self.confidence,
            "reason": self.reason,
            "page_numbers": list(self.page_numbers),
            "original_terminology": self.original_terminology,
            "values": [v.model_dump() for v in self.values],
            "list_value": list(self.list_value),
            "text_value": self.text_value,
            "source_text": self.source_text,
            "source_location": self.source_location.model_dump()
            if self.source_location is not None
            else None,
        }
        if self.has_bool():
            data["bool_value"] = self.bool_or_none()
        return cast(ParameterResultWire, data)


class ProcessDiagnostics(BaseModel):
    """Engine-authored process signals for one extraction run.

    Never populated from model output -- see ``drop_engine_authored_fields``
    (ingest guard) and the shared schema builder in ``agent.py`` (schema
    guard). ``flags`` carries values like ``["tool_bypass"]``; ``tools_called``
    is the ordered list of tool names the agent invoked before submitting.
    """

    model_config = {"extra": "forbid"}

    flags: list[str] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Full extraction result for a PDF datasheet."""

    pdf_source: str
    results: list[ParameterResult]
    parts: list[str] = Field(default_factory=list)
    process_diagnostics: ProcessDiagnostics = Field(default_factory=ProcessDiagnostics)


def normalize_parameters(
    params: Sequence[str | ParameterSpec],
) -> list[ParameterSpec]:
    """Convert a mixed list of strings and ParameterSpecs to all ParameterSpecs."""
    result = []
    for p in params:
        if isinstance(p, str):
            result.append(ParameterSpec(name=p))
        else:
            result.append(p)
    return result


# Fields populated by the extraction engine, never accepted from model output.
ENGINE_AUTHORED_FIELDS: Final[frozenset[str]] = frozenset(
    {"process_diagnostics", "source_location"}
)


def drop_engine_authored_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Strip engine-authored keys from a model-supplied payload, recursively.

    Engine-authored fields (``process_diagnostics`` at the top level,
    ``source_location`` nested under ``results[]`` and ``results[].values[]``)
    are computed by the engine, never accepted from the model. Stripping them at
    every depth before validation stops a fabricated value surviving when the
    gateway does not enforce the output schema. Mutates ``data`` in place and
    returns it for convenience.

    Strips by key name at every depth; the ENGINE_AUTHORED_FIELDS names are
    deliberately engine-only and must never be model-facing field names.
    """

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            # Pop before iterating: keeps the dict size stable during the
            # .values() loop below. Do not reorder these two loops.
            for field in ENGINE_AUTHORED_FIELDS:
                node.pop(field, None)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(data)
    return data
