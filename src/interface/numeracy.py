"""Numeracy screener, administered BEFORE condition assignment.

Instrument: the three-item numeracy scale of Schwartz, Woloshin, Black & Welch
(1997), Annals of Internal Medicine 127(11), 966-972. Short, widely used on
online panels, and it tests the same proportional reasoning the ordering task
needs (turning a risk percentage into an expected count).

Pass mark: 2 of 3 correct. This threshold is a researcher choice and is
registered in docs/OSF-PREREGISTRATION.md section 8; change it there first.

Why before assignment: conditions alternate by arrival order (app.api_start).
If a screened-out person were assigned first, every later participant's
condition would shift and the cells could unbalance. So a failed screener
creates no participant row at all -- only an anonymous count in `screening`.
"""

from __future__ import annotations

PASS_MARK = 2

# id, question as displayed, correct answer, absolute tolerance
ITEMS = (
    ("num_1",
     "Imagine that we roll a fair, six-sided die 1,000 times. Out of 1,000 "
     "rolls, how many times do you think the die would come up even (2, 4, or 6)?",
     500.0, 0.0),
    ("num_2",
     "In a lottery, the chance of winning a 10-dollar prize is 1%. What is your "
     "best guess about how many people would win a 10-dollar prize if 1,000 "
     "people each buy a single ticket?",
     10.0, 0.0),
    ("num_3",
     "In a sweepstakes, the chance of winning a car is 1 in 1,000. What percent "
     "of tickets win a car? (answer in %)",
     0.1, 1e-9),
)

ITEM_IDS = tuple(i[0] for i in ITEMS)


def public_items() -> list[dict]:
    """What the browser may see: questions, never answers."""
    return [{"id": i, "text": t} for i, t, _, _ in ITEMS]


def score(answers: dict) -> int:
    """Number correct. Missing or non-numeric answers count as wrong."""
    if set(answers) != set(ITEM_IDS):
        raise ValueError("expected exactly the numeracy items")
    n = 0
    for item_id, _, correct, tol in ITEMS:
        raw = str(answers[item_id]).strip().replace(",", "").rstrip("%").strip()
        try:
            val = float(raw)
        except ValueError:
            continue
        if abs(val - correct) <= tol:
            n += 1
    return n


def passed(n_correct: int) -> bool:
    return n_correct >= PASS_MARK
