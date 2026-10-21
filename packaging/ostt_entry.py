"""PyInstaller entry point for ostt.exe."""

import sys

from offline_stt.cli import main

if __name__ == "__main__":
    sys.exit(main())
