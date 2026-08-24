"""Shared figure style for both projects.

Two rules do the work here.

Layout: captions are placed BELOW the figure box (negative y in figure
coordinates) and the file is written with bbox_inches="tight", so matplotlib
expands the saved bounding box to contain every artist. That is what stops
suptitles being clipped and stops captions landing on tick labels, which are the
two problems that come from mixing tight_layout with figure-level text.

Palette: an editorial rather than a default-matplotlib look. White ground, one
dominant blue, burnt orange as the accent that marks whatever the figure is
arguing about, and teal as a third categorical color used sparingly. White
rather than a tinted ground, because a figure has to survive being dropped into
a paper, a slide, and a web page without carrying one of those backgrounds with
it.

Text: every piece of text in every figure is black. Titles, axis labels, tick
labels, annotations and source notes all render at full strength. Grey type is
the default in most plotting styles and it is the first thing to disappear when
a figure is printed, projected, or photocopied, so none is used here. Grey
survives only as chart furniture, meaning gridlines and axis rules, where it is
supposed to recede.
"""

import matplotlib.pyplot as plt

# Core palette. BLACK is the ink for every character in every figure.
BLACK = "#000000"
SLATE = BLACK          # kept as an alias: nothing is slate-grey any more

PAPER = "#ffffff"

# Chart furniture only. Never used for text.
GRID = "#e4e7ea"
RULE = "#9ba3ac"

# Primary, accent, and the muted third.
BLUE = "#1c5d99"
ORANGE = "#d1592a"
TEAL = "#12857f"
# GREY is a series color for data, not a text color.
GREY = "#5b6670"

# Tints of the primary, light to dark, for ordered categories.
LIGHT = "#bcd3e8"
MID = "#6fa3c9"
DARK = "#123f69"
AMBER = "#e08a3c"

# Backwards-compatible aliases used by the analysis scripts.
INK = SLATE

# Ordered low -> high income. Runs warm to cool, so the direction of the
# gradient is legible even without reading the axis labels.
INCOME_COLORS = [ORANGE, AMBER, MID, BLUE]

# Unordered categories, maximally separable.
CATEGORY_COLORS = [BLUE, ORANGE, TEAL, AMBER, MID, GREY]


def use_paper_style():
    """Consistent, print-ready defaults for every figure in the project."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.16,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        # Titles run larger and heavier than matplotlib's default. The figure
        # should carry its own argument at a glance.
        "axes.titlesize": 13.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": BLACK,
        "axes.titlelocation": "left",
        "axes.titlepad": 15,
        "axes.labelsize": 11.5,
        "axes.labelweight": "medium",
        "axes.labelcolor": BLACK,
        "axes.edgecolor": RULE,
        "axes.linewidth": 1.0,
        "text.color": BLACK,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.prop_cycle": plt.cycler(color=CATEGORY_COLORS),
        "figure.constrained_layout.use": False,
        # Panels need real air between them; the default 0.2 lets a long
        # y-tick label from one panel land inside its neighbour.
        "figure.subplot.wspace": 0.42,
        "figure.subplot.hspace": 0.34,
    })


def style_axis(ax, axis="y"):
    """Light grid behind the data, no top or right spine."""
    if axis in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    if axis in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    return ax


def _scaled(fig, at_ten_inches, lo, hi):
    """Point size that holds a constant apparent size across figure widths.

    A figure is written at a fixed pixel width regardless of how many inches
    wide it is declared to be, so a 9pt caption on an 18-inch three-panel figure
    renders about half the apparent size of the same 9pt on a 10-inch one. Sizes
    are therefore specified for a 10-inch figure and scaled from there, clamped
    so nothing becomes absurd at the extremes.
    """
    return min(hi, max(lo, at_ten_inches * fig.get_figwidth() / 10.0))


def suptitle(fig, text, size=None):
    """Figure title, stored now and positioned at save time.

    Positioning is deferred because the correct x is the left edge of the
    plotted area including its tick labels, and that is not known until the
    figure has been drawn. Anchoring to the figure edge instead is what made
    figures look right-shifted with dead space on the left: the title started at
    x = 0 while the axes started at the default left margin of 0.125, and the
    tight bounding box then cropped to the title.
    """
    if size is None:
        size = _scaled(fig, 14, 13, 22)
    t = fig.text(0.0, 1.0, text, fontsize=size, fontweight="bold",
                 ha="left", va="bottom", color=BLACK)
    fig._fs_suptitle = t
    return t


def caption(fig, text, size=None):
    """Source note placed below the plotted area, in black.

    Like the title, this is positioned at save time. It is also re-wrapped there
    to the measured width of the plotted area, so the note spans the figure
    rather than stopping wherever the source string happened to contain a
    newline. Any newlines in `text` are treated as spaces for that reason.
    """
    if size is None:
        size = _scaled(fig, 10.2, 9.5, 16)
    t = fig.text(0.0, 0.0, text.replace("\n", " "), fontsize=size,
                 ha="left", va="top", color=BLACK, linespacing=1.45)
    fig._fs_caption = t
    return t


def _content_box(fig, renderer):
    """Bounding box of every axes and its decorations, in figure coordinates.

    This is the region a reader perceives as "the figure", which is what the
    title and caption should align to.
    """
    boxes = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        try:
            boxes.append(ax.get_tightbbox(renderer))
        except Exception:
            boxes.append(ax.get_window_extent(renderer))
    # Figure-level legends sit outside every axes, so they have to be measured
    # separately or the caption is placed on top of them.
    for leg in getattr(fig, "legends", []):
        try:
            boxes.append(leg.get_window_extent(renderer))
        except Exception:
            pass
    if not boxes:
        return None
    from matplotlib.transforms import Bbox
    union = Bbox.union(boxes)
    return union.transformed(fig.transFigure.inverted())


def _rule(fig, x0, x1, y, lw=0.9):
    """Hairline in figure coordinates, drawn outside any axes.

    Used to give the title a baseline and the caption a header rule. Figure-level
    lines rather than axes decorations, because they have to span the whole
    plotted area including panels that do not share an axis.
    """
    from matplotlib.lines import Line2D
    line = Line2D([x0, x1], [y, y], transform=fig.transFigure,
                  color=RULE, linewidth=lw, zorder=0)
    line.set_clip_on(False)
    fig.add_artist(line)
    return line


def _layout_title_and_caption(fig):
    """Align the title and caption to the plotted area and wrap the caption."""
    import textwrap

    t = getattr(fig, "_fs_suptitle", None)
    c = getattr(fig, "_fs_caption", None)
    if t is None and c is None:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = _content_box(fig, renderer)
    if box is None:
        return

    fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
    # Gaps are specified in inches so they do not scale with figure size.
    gap_above = 0.40 / fig_h
    gap_below = 0.42 / fig_h

    if t is not None:
        t.set_position((box.x0, box.y1 + gap_above))
        t.set_ha("left")
        t.set_va("bottom")
        # Hairline under the figure title, spanning the plotted area. Gives the
        # title a baseline to sit on and separates it from the panel titles
        # underneath, which are set in the same family at a smaller size.
        _rule(fig, box.x0, box.x1, box.y1 + gap_above * 0.55)

    if c is not None:
        # Wrap to the wider of the plotted area and the figure itself, so the
        # note runs the full length of the image. Wrapping to a narrow single
        # panel turns a three-line caption into a ten-line block and leaves a
        # nearly square image.
        width_in = max((box.x1 - box.x0) * fig_w, 0.94 * fig_w)
        # DejaVu Sans averages close to 0.52 em per character; at 72 points to
        # the inch that gives the usable characters per line.
        chars = max(40, int(width_in * 72.0 / (c.get_fontsize() * 0.52)))
        c.set_text("\n".join(textwrap.wrap(c.get_text().replace("\n", " "),
                                           width=chars)))
        # Hairline above the caption, so the source note reads as a footer
        # rather than as loose text under the axis labels.
        _rule(fig, box.x0, box.x1, box.y0 - gap_below * 0.42)

        single = len([a for a in fig.axes if a.get_visible()]) == 1
        if single:
            # The block is centered under the plot, but the lines inside it stay
            # left-aligned; centring every line leaves a ragged, poster-like
            # paragraph rather than a source note.
            c.set_position(((box.x0 + box.x1) / 2, box.y0 - gap_below))
            c.set_ha("center")
            c.set_multialignment("left")
        else:
            c.set_position((box.x0, box.y0 - gap_below))
            c.set_ha("left")
        c.set_va("top")


def panel(ax, letter):
    """Bold panel letter, PNAS style, placed outside the axes."""
    ax.text(-0.02, 1.16, letter, transform=ax.transAxes, fontsize=14.5,
            fontweight="bold", va="top", ha="right", color=BLACK)


def bar_labels(ax, values, fmt="{:.2f}", orient="v", pad_frac=0.02,
               inside_above=None, fontsize=10.5):
    """Label bars, putting the text inside when the bar is long enough.

    `inside_above` is a value on the data scale: bars at or beyond it get a
    white label inside the bar, which is what prevents long labels running off
    the end of the axis.
    """
    lo, hi = (ax.get_ylim() if orient == "v" else ax.get_xlim())
    span = hi - lo
    for i, v in enumerate(values):
        inside = inside_above is not None and abs(v) >= inside_above
        off = -pad_frac * span if inside else pad_frac * span
        if orient == "v":
            ax.text(i, v + off, fmt.format(v), ha="center",
                    va="top" if inside else "bottom",
                    color="white" if inside else BLACK,
                    fontsize=fontsize, fontweight="semibold", zorder=8)
        else:
            ax.text(v + off, i, fmt.format(v),
                    ha="right" if inside else "left", va="center",
                    color="white" if inside else BLACK,
                    fontsize=fontsize, fontweight="semibold", zorder=8)


def save(fig, path, wide=False):
    """Write PNG and PDF. bbox_inches='tight' is set globally in rcParams.

    The title and caption are positioned here rather than where they were
    declared, because aligning them to the plotted area requires a drawn figure.
    """
    _layout_title_and_caption(fig)
    fig.savefig(path, format="png")
    fig.savefig(str(path).replace(".png", ".pdf"), format="pdf")
    plt.close(fig)


def compose(panel_paths, titles, out_path, title, note, figsize=None):
    """Combine already-rendered panel images into one labeled figure.

    Some plotting libraries -- SHAP's summary_plot in particular -- resize and
    re-position the active figure as a side effect, which leaves figure-level
    titles and captions outside the saved bounding box. Rendering each panel to
    its own file first and compositing here sidesteps that entirely.

    Panels rarely come back the same shape, so the column widths are set from
    each image's own aspect ratio and every panel is given the same height. That
    is what keeps a composed figure from looking like one wide panel beside one
    narrow one with a gap of white between them.
    """
    import matplotlib.image as mpimg
    # Reset the style first: some libraries (SHAP) mutate rcParams as a side
    # effect, which can leave figure text rendering in the wrong color.
    use_paper_style()

    images = [mpimg.imread(str(path)) for path in panel_paths]
    aspects = [im.shape[1] / im.shape[0] for im in images]   # width / height

    if figsize is None:
        panel_h = 5.0
        width = sum(a * panel_h for a in aspects) + 0.6 * (len(images) - 1)
        # Floor as well as ceiling: too narrow and the caption wraps to a wall
        # of text taller than the panels it describes.
        width = max(12.5, min(width, 16.4))
        figsize = (width, width / sum(aspects) + 0.6)

    fig, axes = plt.subplots(1, len(images), figsize=figsize,
                             gridspec_kw={"width_ratios": aspects,
                                          "wspace": 0.10})
    if len(images) == 1:
        axes = [axes]
    for ax, im, panel_title in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_title(panel_title, fontsize=12.5, loc="left", pad=10,
                     fontweight="bold", color=BLACK)
        ax.axis("off")

    suptitle(fig, title)
    caption(fig, note)
    # compose() writes the file itself rather than calling save(), so the
    # deferred title and caption layout has to be run explicitly here.
    _layout_title_and_caption(fig)
    fig.savefig(out_path, format="png")
    fig.savefig(str(out_path).replace(".png", ".pdf"), format="pdf")
    plt.close(fig)
