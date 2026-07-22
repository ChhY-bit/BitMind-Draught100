from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# CJK font registration — multi-backend, works across Wayland / Xorg /
# macOS / Windows without user intervention.
# ---------------------------------------------------------------------------

# Font families we try, ordered by preference.  The list covers:
#   • fontconfig-managed fonts (Tk built with Xft on Linux)
#   • the bundled TTF installed into fontconfig / X11 path at startup
#   • X11 core bitmaps (last resort — limited encoding, may render poorly)
#   • Windows / macOS system fonts
_PREFERRED_CJK_FONTS = [
    # fontconfig families (Xft-capable Tk on Linux)
    'noto sans cjk sc', 'noto sans cjk tc', 'noto sans cjk jp',
    'ar pl uming cn', 'ar pl ukai cn',
    'wenquanyi micro hei', 'wenquanyi zen hei',
    # bundled font — family name as registered via fontconfig *or* X11
    'droid sans fallback',
    # Windows
    'microsoft yahei', 'simhei',
    # macOS
    'pingfang sc', 'heiti sc',
]

# X11 core bitmap CJK fonts — only used when nothing else is available.
# These use legacy encodings (GB2312) which may show garbled glyphs for
# characters outside the encoding's repertoire.
_FALLBACK_CJK_FONTS = ['song ti', 'fangsong ti']

# Track resources for cleanup on exit.
_registered_font_dir: Optional[str] = None
_fontconfig_font_dir: Optional[str] = None

# Cached path to bundled TTF — set by _get_bundled_ttf_path().
_BUNDLED_TTF_PATH: Optional[str] = None


def _get_bundled_ttf_path() -> Optional[str]:
    """Return the path to the bundled CJK TTF, or None."""
    global _BUNDLED_TTF_PATH
    if _BUNDLED_TTF_PATH is None:
        candidate = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'assets', 'fonts', 'DroidSansFallbackFull.ttf',
        )
        if os.path.exists(candidate):
            _BUNDLED_TTF_PATH = candidate
    return _BUNDLED_TTF_PATH


# ---------------------------------------------------------------------------
# Dynamic system CJK font discovery
# ---------------------------------------------------------------------------


def _discover_system_cjk_ttfs() -> list[str]:
    """Find CJK TrueType/OpenType font files installed on the system.

    Uses ``fc-list`` (fontconfig) for cross-distro discovery, with
    hardcoded fallback paths for minimal environments.

    Returns file paths ordered by preference (sans-serif families first,
    then serif) so the smoothest screen fonts are tried earliest.
    """
    discovered: list[str] = []

    # ── Dynamic discovery via fontconfig (works on all Linux distros) ──
    try:
        result = subprocess.run(
            ['fc-list', ':lang=zh', 'file'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                # Format: /path/to/font.ttc: family,family:style=Regular
                filepath = line.rsplit(':', 1)[0].strip()
                if filepath and os.path.exists(filepath):
                    discovered.append(filepath)
    except Exception:
        pass

    # ── Fallback: walk common system font directories ──
    if not discovered:
        for search_dir in [
            '/usr/share/fonts/truetype/wqy',
            '/usr/share/fonts/opentype/noto',
            '/usr/share/fonts/truetype/noto',
            '/usr/share/fonts/truetype/arphic',
            '/usr/share/fonts/opentype',
            '/usr/share/fonts/truetype',
            '/usr/local/share/fonts',
        ]:
            if not os.path.isdir(search_dir):
                continue
            for dirpath, _dirs, fnames in os.walk(search_dir):
                for fname in fnames:
                    if fname.lower().endswith(('.ttf', '.ttc', '.otf')):
                        discovered.append(os.path.join(dirpath, fname))

    # ── Rank: sans-serif before serif (smoother at small sizes) ──
    _SANS_FAMILIES = (
        'wqy-microhei', 'wenquanyi micro hei', 'wenquanyi zen hei',
        'notosans', 'noto sans',
        'sourcehansans', 'source han sans',
        'microsoft yahei', 'simhei',
    )

    def _rank(p: str) -> tuple[int, int, int, str]:
        basename = os.path.basename(p).lower()
        is_serif = 0 if any(k in basename for k in _SANS_FAMILIES) else 1
        is_wqy = 0 if 'wqy-microhei' in basename else 1
        is_regular = 0 if any(w in basename for w in ('regular', 'medium', 'micro')) else 1
        return (is_serif, is_wqy, is_regular, basename)

    ranked = sorted(set(discovered), key=_rank)
    return ranked[:5]


# ---------------------------------------------------------------------------
# Font installation helpers
# ---------------------------------------------------------------------------


def _get_tk_font_families(root: tk.Tk) -> set:
    try:
        return set(root.tk.call('font', 'families'))
    except Exception:
        return set()


def _install_fontconfig(bundled_ttf: str) -> Optional[str]:
    """Copy the bundled TTF into the user fontconfig directory and run
    ``fc-cache``.  Returns the font family name on success.
    """
    global _fontconfig_font_dir

    if not sys.platform.startswith('linux'):
        return None

    font_dir = os.path.join(
        os.environ.get('XDG_DATA_HOME', os.path.join(str(Path.home()), '.local', 'share')),
        'fonts',
    )
    os.makedirs(font_dir, exist_ok=True)

    dest = os.path.join(font_dir, os.path.basename(bundled_ttf))
    if not os.path.exists(dest):
        try:
            shutil.copy2(bundled_ttf, dest)
        except OSError:
            return None

    # Refresh fontconfig cache (ignore failures — font may still work)
    try:
        subprocess.run(
            ['fc-cache', '-f', font_dir],
            capture_output=True, timeout=30,
        )
    except Exception:
        pass

    _fontconfig_font_dir = font_dir

    # Parse the TTF to extract the font family name using fc-scan
    try:
        result = subprocess.run(
            ['fc-scan', '--format=%{family}', dest],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            family = result.stdout.strip().split(',')[0].strip()
            return family
    except Exception:
        pass

    return None


def _install_x11_font(source_path: str) -> Optional[str]:
    """Register a TTF/TTC file in the X11 core font path (``xset +fp``).

    This is the fallback for Tk builds **without** Xft (e.g. conda's
    ``tk-noxft`` package).  It requires ``mkfontscale`` / ``mkfontdir``
    and a running X server that accepts font-path changes.
    """
    global _registered_font_dir

    if not sys.platform.startswith('linux'):
        return None

    font_dir = os.path.join(str(Path.home()), '.bitmind_draughts', 'fonts')
    os.makedirs(font_dir, exist_ok=True)

    dest = os.path.join(font_dir, os.path.basename(source_path))
    if not os.path.exists(dest):
        try:
            shutil.copy2(source_path, dest)
        except OSError:
            return None

    # Generate X11 font metadata
    try:
        subprocess.run(
            ['mkfontscale', font_dir],
            check=True, capture_output=True, timeout=10,
        )
        subprocess.run(
            ['mkfontdir', font_dir],
            check=True, capture_output=True, timeout=10,
        )
    except Exception:
        return None

    # Add to X server font path
    try:
        subprocess.run(
            ['xset', '+fp', font_dir],
            check=True, capture_output=True, timeout=10,
        )
        subprocess.run(
            ['xset', 'fp', 'rehash'],
            check=True, capture_output=True, timeout=10,
        )
    except Exception:
        return None

    _registered_font_dir = font_dir

    # Parse fonts.dir to extract the X11 font family name
    fonts_dir_file = os.path.join(font_dir, 'fonts.dir')
    try:
        with open(fonts_dir_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line[0].isdigit():
                    continue
                idx = line.find(' -')
                if idx < 0:
                    continue
                xlfd = line[idx + 1:].strip()
                fields = xlfd.split('-')
                if len(fields) >= 3:
                    return fields[2]
    except Exception:
        pass

    return None


def _unregister_fonts() -> None:
    """Best-effort cleanup of font registrations on exit."""
    global _registered_font_dir
    if _registered_font_dir is not None:
        try:
            subprocess.run(
                ['xset', '-fp', _registered_font_dir],
                capture_output=True, timeout=5,
            )
            subprocess.run(
                ['xset', 'fp', 'rehash'],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
        _registered_font_dir = None


# ---------------------------------------------------------------------------
# CJK rendering test
# ---------------------------------------------------------------------------


def _cjk_renders_correctly(family: str, root: tk.Tk) -> bool:
    """Verify *family* actually renders CJK glyphs correctly.

    Some font backends (notably XWayland serving TrueType fonts through
    the X11 core protocol) produce corrupted glyph metrics — CJK
    characters get wildly inconsistent widths.  We detect this by
    measuring several CJK glyphs and checking they are all roughly the
    same width (as expected for ideographic characters).

    Also catches fonts that lack CJK entirely (zero-width glyphs).
    """
    try:
        f = tk.font.Font(root=root, family=family, size=14)
        cjk_widths = [f.measure(ch) for ch in '中对局黑白']
        ascii_w = f.measure('A')

        if any(w < ascii_w * 0.6 for w in cjk_widths):
            return False

        median = sorted(cjk_widths)[len(cjk_widths) // 2]
        if median <= 0:
            return False
        if any(abs(w - median) / median > 0.35 for w in cjk_widths):
            return False  # inconsistent → corrupted glyph mapping

        return True
    except Exception:
        return False
