"""Deterministic, source-defined strata for optional field diagnostics."""

from .field_probe import CODE_POSITIONS


def stratified_code_worlds(generator, *, worlds_per_position, language="en"):
    if worlds_per_position < 1:
        raise ValueError("worlds per character position must be positive")
    counts = {position: 0 for position in CODE_POSITIONS}
    selected = []
    for world in range(1000 * worlds_per_position):
        row = generator.episode(world, variant=0, family="single_fact", language=language)
        position = row["annotations"].get("counterfactual_character_position")
        if position not in counts:
            raise ValueError("field-stratified sampling requires v4/v5 code character annotations")
        if counts[position] >= worlds_per_position:
            continue
        selected.append(world)
        counts[position] += 1
        if min(counts.values()) == worlds_per_position:
            return selected
    raise RuntimeError("insufficient independent worlds for all character strata")
