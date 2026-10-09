"""Dispatch immutable episode implementations without editing older revisions."""

import hashlib
from pathlib import Path

from .robust import (AB_DATA_REVISION as AB_V2_REVISION, LEGACY_AB_REVISION,
                     data_identity as v2_identity, make_generator as v2_generator)
from .robust_order import AB_ORDER_REVISION, RandomOrderEpisodeGenerator
from .robust_fields import AB_FIELDS_REVISION, AllFieldsEpisodeGenerator

AB_DATA_REVISION = AB_FIELDS_REVISION
AB_REVISIONS = (LEGACY_AB_REVISION, AB_V2_REVISION, AB_ORDER_REVISION, AB_FIELDS_REVISION)


def make_generator(*, revision=AB_DATA_REVISION, **kwargs):
    if revision == AB_FIELDS_REVISION:
        return AllFieldsEpisodeGenerator(**kwargs)
    if revision == AB_ORDER_REVISION:
        return RandomOrderEpisodeGenerator(**kwargs)
    return v2_generator(revision=revision, **kwargs)


def data_identity(revision):
    if revision not in (AB_ORDER_REVISION, AB_FIELDS_REVISION):
        return v2_identity(revision)
    identity = v2_identity(AB_V2_REVISION)
    identity["revision"] = revision
    path = Path(__file__).resolve().with_name("robust_order.py")
    identity["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if revision == AB_FIELDS_REVISION:
        for name in ("robust_fields.py", "general_keep.py"):
            path = Path(__file__).resolve().with_name(name)
            identity["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identity
