from importlib.metadata import version

from greifer.app import main

__version__ = version("greifer")
__all__ = ["__version__", "main"]
