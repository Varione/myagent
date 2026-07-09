"""UI package exports.

Avoid importing GUI dependencies at module import time so headless tests can
import controller code without requiring CustomTkinter.
"""

__all__ = ["main"]


def main():
    from .app import main as _main

    return _main()
