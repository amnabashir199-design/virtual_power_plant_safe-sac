from __future__ import annotations

import json

from reproduction_core import validate_dataset


if __name__ == "__main__":
    print(json.dumps(validate_dataset(), indent=2))
