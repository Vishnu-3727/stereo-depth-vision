"""Turn measured values into experiment conclusions.

EXP-016 recorded the conclusion "validation improves" while its own validation
EPE went from 18.497 px at epoch 0 to 19.216 px at epoch 19. The claim was
withdrawn in `experiments/EXP-016/CORRECTION.md`, but the sentence was a literal
string in the experiment script, so re-running it would have recreated the false
claim.

This module exists so that a conclusion cannot be written independently of the
data it describes. Every function here takes measurements and returns a
statement about them; none of them can assert an outcome that the numbers do
not support.
"""

from __future__ import annotations

from typing import Iterable, Literal

ValidationChange = Literal["improved", "did not improve", "unchanged"]


def classify_validation_change(
    initial: float,
    final: float,
    lower_is_better: bool = True,
    tolerance: float = 0.0,
) -> ValidationChange:
    """Compare a validation metric at the start and end of a run.

    Returns ``"improved"``, ``"did not improve"`` or ``"unchanged"``.

    The comparison is first recorded value against last recorded value, which is
    the criterion `experiments/EXP-016/CORRECTION.md` uses when it withdraws the
    original claim: it sets epoch 0's 18.497 px against epoch 19's 19.216 px and
    concludes validation did not improve. Best-epoch performance is reported
    alongside as context but is deliberately not the verdict, because a run whose
    best epoch is in the middle and whose final epoch is worse has not
    demonstrated that training improves validation.

    ``lower_is_better`` is explicit rather than assumed: EPE and D1 are error
    metrics where lower is better, but a caller passing an accuracy would need
    the opposite, and silently assuming a direction is the same class of mistake
    this module exists to prevent.

    ``tolerance`` defaults to 0.0, so ``"unchanged"`` means exactly equal. A
    caller that wants a band of indifference must choose and document its width;
    picking one here would invent a scientific criterion the experiment never
    defined.
    """
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative, got " + repr(tolerance))

    delta = final - initial
    if abs(delta) <= tolerance:
        return "unchanged"
    improved = delta < 0 if lower_is_better else delta > 0
    return "improved" if improved else "did not improve"


def classify_series_trend(
    values: Iterable[float], lower_is_better: bool = True, tolerance: float = 0.0
) -> ValidationChange:
    """Same comparison, applied to the first and last of a sequence."""
    series = list(values)
    if len(series) < 2:
        raise ValueError(
            "need at least two values to describe a trend, got "
            + str(len(series))
        )
    return classify_validation_change(
        series[0], series[-1], lower_is_better=lower_is_better, tolerance=tolerance
    )


def describe_validation(
    initial: float,
    final: float,
    best: float | None = None,
    best_label: str = "",
    metric: str = "EPE",
    unit: str = "px",
) -> str:
    """A sentence about a validation series that matches its own numbers."""
    verdict = classify_validation_change(initial, final)
    text = (
        "Validation {} went from {:.3f} {} to {:.3f} {}, so validation "
        "{}.".format(metric, initial, unit, final, unit, verdict)
    )
    if best is not None:
        text += " The best recorded value was {:.3f} {}{}.".format(
            best, unit, (" at " + best_label) if best_label else ""
        )
        if verdict != "improved":
            text += (
                " A better intermediate value does not establish improvement, "
                "since the run did not end there."
            )
    return text


def describe_training_outcome(
    losses: list[float],
    grad_norms: list[float],
    val_initial: float | None = None,
    val_final: float | None = None,
    val_best: float | None = None,
    val_best_label: str = "",
    checkpoint_written: bool = False,
) -> str:
    """Assemble a training conclusion entirely from measured values.

    Nothing here is asserted in advance. If the loss rose, this says so; if a
    gradient was non-finite, this says so; if validation got worse, this says
    that too.
    """
    import math

    parts: list[str] = []

    if len(losses) >= 2:
        loss_trend = classify_series_trend(losses)
        parts.append(
            "Training loss went from {:.4f} to {:.4f}, so the loss {}.".format(
                losses[0], losses[-1],
                {"improved": "decreased", "did not improve": "increased",
                 "unchanged": "was unchanged"}[loss_trend],
            )
        )
    elif losses:
        parts.append("Only one training loss value was recorded: {:.4f}.".format(losses[0]))

    if grad_norms:
        finite = [g for g in grad_norms if math.isfinite(g)]
        non_finite = len(grad_norms) - len(finite)
        if non_finite:
            parts.append(
                "{} of {} gradient norms were non-finite.".format(
                    non_finite, len(grad_norms)
                )
            )
        else:
            parts.append(
                "All {} gradient norms were finite, with a maximum of "
                "{:.3f}.".format(len(grad_norms), max(finite))
            )

    if val_initial is not None and val_final is not None:
        parts.append(
            describe_validation(
                val_initial, val_final, best=val_best, best_label=val_best_label
            )
        )
    else:
        parts.append("No validation measurements were recorded.")

    if checkpoint_written:
        parts.append("A checkpoint was written.")

    return " ".join(parts)


def demo() -> None:
    """Self-check, including the exact numbers EXP-016 actually produced."""
    # EXP-016: 18.497 px at epoch 0, 19.216 px at epoch 19. Validation got worse.
    assert classify_validation_change(18.497, 19.216) == "did not improve"
    assert classify_validation_change(19.216, 18.497) == "improved"
    assert classify_validation_change(18.497, 18.497) == "unchanged"

    # direction is explicit, not assumed
    assert classify_validation_change(0.80, 0.90, lower_is_better=False) == "improved"
    assert classify_validation_change(0.90, 0.80, lower_is_better=False) == "did not improve"

    # a tolerance band must be chosen by the caller
    assert classify_validation_change(10.0, 10.05, tolerance=0.1) == "unchanged"
    assert classify_validation_change(10.0, 10.5, tolerance=0.1) == "did not improve"

    # the sentence agrees with the numbers, and a good middle epoch does not
    # turn a worse ending into an improvement
    text = describe_validation(18.497, 19.216, best=17.622, best_label="epoch 15")
    assert "did not improve" in text, text
    assert "improved." not in text, text
    assert "17.622" in text and "does not establish improvement" in text, text

    # a real improvement is stated plainly
    text2 = describe_validation(19.216, 18.497)
    assert "so validation improved." in text2, text2

    # the assembled outcome reflects what happened in EXP-016
    out = describe_training_outcome(
        losses=[11.4937, 7.6137],
        grad_norms=[42.5, 38.4, 814.2],
        val_initial=18.497,
        val_final=19.216,
        val_best=17.622,
        val_best_label="epoch 15",
        checkpoint_written=True,
    )
    assert "loss decreased" in out, out
    assert "did not improve" in out, out
    assert "All 3 gradient norms were finite" in out, out
    assert "A checkpoint was written." in out, out

    # a diverging run is described as diverging
    bad = describe_training_outcome(
        losses=[1.0, 5.0], grad_norms=[1.0, float("nan")],
        val_initial=1.0, val_final=2.0,
    )
    assert "loss increased" in bad, bad
    assert "1 of 2 gradient norms were non-finite" in bad, bad

    # a series trend needs at least two points
    try:
        classify_series_trend([1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("a single value should not yield a trend")

    print("conclusions self-check passed")


if __name__ == "__main__":
    demo()
