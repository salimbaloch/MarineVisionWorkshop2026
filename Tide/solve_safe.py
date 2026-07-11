#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solve_safe.py -- time-limited clingo solve, drop-in for learn_weights.solve.

learn_weights.solve runs clingo with --opt-mode=opt --models=0 (prove optimum,
enumerate all optimal models). On a hard instance (many candidate atoms, weight
structure that makes the search space huge) that can run effectively forever --
which is what stalls the full-matrix run when the SeaClear-urchin weights are
applied to a different class's facts.

solve_timed() caps the wall-clock per solve. clingo keeps improving the best
model during optimization, so on timeout we return the best-so-far -- a
good-enough explanation rather than a hang. Signature matches learn_weights.solve
((atoms, cost)) so it is a drop-in; solve_timed additionally returns a timeout
flag for callers that want it.

Use WITHOUT editing box_regressor.py or learn_weights.py -- monkeypatch at import:

    import box_regressor, solve_safe
    box_regressor.solve = lambda prog: solve_safe.solve_timed(prog, 5.0)[:2]

(eval_coco_ap.py does exactly this when --solve-timeout is set.)
"""
import clingo


def solve_timed(program, time_limit=5.0):
    """Best model within time_limit seconds.
    Returns (atoms:set[str], cost:list|None, timed_out:bool).
    On timeout returns the best model found so far (may be empty if none found
    yet, in which case the caller treats it like a failed solve)."""
    ctl = clingo.Control(["--opt-mode=opt", "--models=0", "--warn=none"])
    ctl.add("base", [], program)
    ctl.ground([("base", [])])
    best = {"atoms": set(), "cost": None}

    def on_model(m):
        best["atoms"] = {str(s) for s in m.symbols(shown=True)}
        best["cost"] = list(m.cost)

    with ctl.solve(on_model=on_model, async_=True) as handle:
        finished = handle.wait(time_limit)
        if not finished:
            handle.cancel()
    return best["atoms"], best["cost"], (not finished)


def solve(program, time_limit=5.0):
    """Exact drop-in for learn_weights.solve (returns (atoms, cost))."""
    a, c, _ = solve_timed(program, time_limit)
    return a, c


if __name__ == "__main__":
    # tiny self-test: a trivially-solvable program returns fast, not timed out.
    prog = "a. b. {c} :- a. #maximize {1,c:c}. #show c/0."
    atoms, cost, to = solve_timed(prog, 5.0)
    print("atoms:", atoms, "cost:", cost, "timed_out:", to)
    assert not to, "trivial program should not time out"
    print("solve_safe self-test: PASS")