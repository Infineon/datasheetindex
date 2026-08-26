"""No shipped data file may carry a person's name in its ``annotator:`` field.

Two real annotators' names reached a public release this way -- one in
``rederivation.*.yaml``, one in ``classifier_gold.yaml`` -- because the
skeleton generators asked for "your name" and ``data/`` ships with the
benchmark.

The check is deliberately a POSITIVE one: an ``annotator:`` value must match
``_ALLOWED``, i.e. be an ``annotator-N`` label, an ``llm-prescreen`` marker, or
empty. A blocklist of the two names that leaked would pass for every name that
has not leaked yet, which is the only case that matters.

It also fails on the generator prompt that caused the leak, so the wording
cannot come back and re-teach the next annotator to type their name.
"""

from __future__ import annotations

import re

import pytest
import yaml

from chamberbench.claimsio import BENCHMARK_ROOT, data_dir

# `annotator-1`, `annotator-2`, ... | the LLM pre-screen | not yet labelled.
_ALLOWED = re.compile(r"^(annotator-\d+|llm-prescreen(\b.*)?|)$")

_GENERATORS = ("prepare_rederivation.py", "prepare_gold_labelling.py")

# The prompt that produced both leaks, in the wording the generators used.
_NAME_PROMPT = re.compile(r"FILL IN:\s*your name", re.IGNORECASE)


def _data_files() -> list:
    return sorted(data_dir().glob("*.yaml"))


def test_there_are_data_files_to_check():
    """A glob that matches nothing would make every check below vacuous."""
    assert _data_files(), f"no *.yaml under {data_dir()}"


@pytest.mark.parametrize("path", _data_files(), ids=lambda p: p.name)
def test_annotator_field_is_a_non_identifying_label(path):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        return
    metadata = doc.get("metadata")
    if not isinstance(metadata, dict) or "annotator" not in metadata:
        return
    value = str(metadata.get("annotator") or "").strip()
    assert _ALLOWED.match(value), (
        f"{path.name}: annotator {value!r} is not a non-identifying label. "
        "Use annotator-1, annotator-2, ... -- never a person's name; "
        "data/ ships with the benchmark."
    )


@pytest.mark.parametrize("name", _GENERATORS)
def test_generator_does_not_ask_the_annotator_for_a_name(name):
    """The skeletons an annotator fills in must not prompt for a real name."""
    text = (BENCHMARK_ROOT / "scripts" / name).read_text(encoding="utf-8")
    assert not _NAME_PROMPT.search(text), (
        f"scripts/{name} still asks the annotator for their name; "
        "ask for a non-identifying label such as annotator-1 instead."
    )


@pytest.mark.parametrize("path", _data_files(), ids=lambda p: p.name)
def test_data_file_does_not_ask_the_annotator_for_a_name(path):
    assert not _NAME_PROMPT.search(path.read_text(encoding="utf-8")), (
        f"{path.name} still carries the 'FILL IN: your name' prompt."
    )
