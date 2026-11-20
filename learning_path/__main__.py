"""Module execution support: python -m learning_path ..."""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
