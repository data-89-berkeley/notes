import html
from pathlib import Path
from IPython.display import HTML

SET_EXPLORER_PATH = Path(__file__).parent / "set_explorer.html"


def show_set_explorer(height=780):
    """Embed the deck-of-cards Set Explorer (a self-contained HTML/JS page).

    The page is inlined with ``srcdoc`` so it runs in the browser without a kernel,
    both in Jupyter and on the static course-notes site. The frame resizes itself
    to fit its content; ``height`` is only the initial height.
    """
    src = SET_EXPLORER_PATH.read_text(encoding="utf-8")
    return HTML(
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<title>Set Explorer</title></head><body>'
        f'<div class="set-explorer"><iframe title="Set Explorer: events in a deck of 52 cards" '
        f'srcdoc="{html.escape(src, quote=True)}" '
        f'style="width: 100%; height: {height}px; border: none;"></iframe></div>'
        '</body></html>'
    )
