"""Episode schemas, deterministic generation and native-template compilation."""

__all__ = ["EpisodeCompiler", "CompiledEpisode", "SupervisedSequence"]


def __getattr__(name):
    if name in __all__:
        from . import compiler
        return getattr(compiler, name)
    raise AttributeError(name)
