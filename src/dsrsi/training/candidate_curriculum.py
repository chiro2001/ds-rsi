"""Versioned, paired-world single/double memory rehearsal for A facts."""

import hashlib


SINGLE_DOUBLE_V1 = "single-double-world-v1"


def candidate_count(*, seed, ordinal, family, maximum, curriculum=None):
    if curriculum is None:
        return maximum
    if curriculum != SINGLE_DOUBLE_V1 or maximum != 2 or ordinal < 0:
        raise ValueError("unsupported single/double candidate curriculum")
    # Both variants of a counterfactual world have the same candidate count.
    # Keep episodes retain the original two-candidate interference condition.
    if family != "single_fact":
        return maximum
    key = f"{SINGLE_DOUBLE_V1}:{seed}:{ordinal // 2}"
    return 1 + int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % 2
