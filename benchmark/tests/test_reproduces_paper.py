"""The archive still reproduces the paper's headline numbers.

This is the test that gives the release its meaning. Everything else here
checks that a function behaves; this checks that the *shipped artifacts* --
the claim set, the archived model outputs, and the grading surface applied to
them -- still yield the numbers a reader can look up in the paper.

It is deliberately written against the package API rather than by parsing a
script's stdout, so that reformatting a report cannot break it and, more
importantly, cannot silently hide a change in the numbers themselves.

If one of these fails, the release has drifted from the paper. That is a
finding to publish, not a test to update.
"""

from __future__ import annotations

import json

import pytest

from chamberbench.claimsio import archive_dir, load_claims
from chamberbench.variance import aggregate_variance

# Paper, Section 4: the claim set and the off-corpus set.
EXPECTED_CLAIM_COUNT = 25
EXPECTED_A4988_CLAIM_COUNT = 12

# Paper, Table 1 and Section 4. Per-repeat pass counts are pinned alongside the
# mean because the mean alone cannot distinguish a stable model from one whose
# failures happen to average out -- which is the entire Qwen finding.
EXPECTED_FIDELITY = {
    "claudesonnet4.6": {
        "per_run": [25, 25, 25],
        "mean": 25.0,
        "std": 0.0,
        "stable": 25,
    },
    "gpt-5.1": {"per_run": [25, 25, 25], "mean": 25.0, "std": 0.0, "stable": 25},
    "qwen3.6-27b": {"per_run": [23, 19, 15], "mean": 19.0, "std": 4.0, "stable": 13},
}


def _variance():
    path = archive_dir() / "variance_chamber.json"
    if not path.exists():
        pytest.skip("variance_chamber.json not present in the archive")
    return json.loads(path.read_text(encoding="utf-8"))


def test_claim_set_size():
    assert len(load_claims()) == EXPECTED_CLAIM_COUNT
    assert len(load_claims("claims_a4988.yaml")) == EXPECTED_A4988_CLAIM_COUNT


@pytest.mark.parametrize("model", sorted(EXPECTED_FIDELITY))
def test_fidelity_matches_the_paper(model):
    agg = aggregate_variance(_variance()["runs"])
    assert model in agg, f"{model} missing from the archived runs"
    got, want = agg[model]["fidelity"], EXPECTED_FIDELITY[model]
    assert got["per_run"] == want["per_run"]
    assert got["mean"] == pytest.approx(want["mean"], abs=0.05)
    assert got["std"] == pytest.approx(want["std"], abs=0.05)


@pytest.mark.parametrize("model", sorted(EXPECTED_FIDELITY))
def test_claim_stability_matches_the_paper(model):
    """25/25 for the frontier models, 13/25 for Qwen.

    The gap is the paper's reproducibility-of-the-agent result, and it is the
    number most likely to move silently if the archive is regenerated.
    """
    agg = aggregate_variance(_variance()["runs"])
    stability = agg[model]["claim_stability"]
    n_stable = (
        stability["n_stable"] if "n_stable" in stability else stability.get("stable")
    )
    assert n_stable == EXPECTED_FIDELITY[model]["stable"]


def test_reproducibility_verdict_never_sees_agent_output():
    """The structural independence the paper claims, asserted on the signature.

    A future refactor that threads an extraction into `verdict()` would make
    the two axes non-independent while every numeric test above still passed.
    """
    import inspect

    from chamberbench.reproducibility import verdict

    params = set(inspect.signature(verdict).parameters)
    assert params == {"claim", "measurement"}, (
        f"verdict() takes {params}; it must see only the claim and the physical "
        "measurement, never agent output"
    )
