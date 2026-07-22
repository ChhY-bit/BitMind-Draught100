from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — ensure interface/ and project root are on sys.path before
# the gui package tries to import draughts_env or draughts.*.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_INTERFACE_DIR = _PROJECT_ROOT / "interface"
if str(_INTERFACE_DIR) not in sys.path:
    sys.path.insert(0, str(_INTERFACE_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gui.app import main  # noqa: E402

if __name__ == "__main__":
    main()
