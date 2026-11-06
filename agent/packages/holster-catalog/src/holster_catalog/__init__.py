"""Holster's catalog core: the TMDB client and the agent over it.

Import from the submodules. Nothing is re-exported here, so importing the TMDB
client alone does not load LangChain."""

from importlib.metadata import version

__all__ = ["__version__"]

__version__ = version("holster-catalog")
