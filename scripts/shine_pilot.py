#!/usr/bin/env python3
from dsrsi.training.shine_pilot import main


if __name__ == "__main__":
    from dsrsi.evaluation.journal import EvaluationPaused
    try:
        main()
    except EvaluationPaused as exc:
        print(str(exc),flush=True)
