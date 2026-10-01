"""Figure 1 (pipeline), in the paper's notation (T, T~, T^, D_phi, r_keep) and Times. The example strings are
verbatim from the BBC workbook, sheet wf_6classes, row 0, columns Articles / del_zipf_50 / polished_50; do not
prettify them. Output: pipeline_figure.pdf (+ .png preview).
"""
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"],
                            "mathtext.fontset": "stix", "pdf.fonttype": 42})
HERE = Path(__file__).parent
ORIG = ("Choose hope over fear - Kennedy..Voters will\nhave a clear choice between the politics of\n"
        "fear and the politics of hope in the next general\nelection, said Charles Kennedy...In his New\n"
        "Year message the Liberal Democrat leader said\nLabour and the Conservatives were united in\nrelying on fear...")
SKEL = ("Choosehope  fear  ..Voters have\nchoice  of fearand the politicsof\ngeneral Kennedy..In Year message\n"
        "the  Labour Conservatives\nunitedinrelyingfear  .")   # wrapped at spaces only
RECON = ("Choose hope over fear. Voters have\na choice between fear and the politics\nof general Kennedy. In his New Year\n"
         "message, Labour and Conservatives\nunited in relying on fear...")
INK, MUTED, ACC = "#1a1a2e", "#5a6472", "#3b6fd4"
BOX_TEXT, BOX_PROC, EDGE_TEXT = "#f4f6fb", "#dbe7ff", "#8a94a6"

def box(ax, x, y, w, h, title, body, fc, ec, title_size=8.5, body_size=7.0, mono=False, body_dy=0.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=INK, linespacing=1.1)
    if body:
        ax.text(x + w / 2, y + h - 0.36 + body_dy, body, ha="center", va="top", fontsize=body_size, color=INK,
                family="monospace" if mono else "serif", linespacing=1.22)

def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=INK))

def main():
    fig, ax = plt.subplots(figsize=(6.3, 0.98))
    ax.set_xlim(-0.006, 1.006); ax.set_ylim(-0.02, 1.02); ax.axis("off")
    y0, h = 0.02, 0.96
    box(ax, 0.0, y0, 0.275, h, "Source text $T$", ORIG, BOX_TEXT, EDGE_TEXT, title_size=7.4, body_size=5.4, body_dy=0.0)
    arrow(ax, 0.280, 0.306, y0 + h / 2)
    box(ax, 0.310, y0, 0.135, h, "Encoder: deletion",
        "Step / WordLen /\nWordFreq / LP /\nEntropy / Hybrid\n(interchangeable)", BOX_PROC, ACC, title_size=7.4, body_size=5.9, body_dy=0.0)
    arrow(ax, 0.450, 0.476, y0 + h / 2)
    box(ax, 0.480, y0, 0.265, h, r"Skeleton $\tilde{T}$ (still text)", SKEL, BOX_TEXT, EDGE_TEXT, mono=True, title_size=7.4, body_size=5.4, body_dy=0.0)
    arrow(ax, 0.750, 0.776, y0 + h / 2)
    box(ax, 0.780, y0, 0.220, h, r"LLM decoder $D_\phi$: $\hat{T}$", RECON, BOX_TEXT, EDGE_TEXT, title_size=7.4, body_size=5.7, body_dy=0.0)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(HERE / "pipeline_figure.pdf"); fig.savefig(HERE / "pipeline_figure.png", dpi=200)
    print("wrote pipeline_figure.pdf/png")

if __name__ == "__main__":
    main()
