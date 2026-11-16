# Draws the example figures (correct / incorrect answers) from the examples folder written by clip_t5.py.
# Usage: python scripts/make_example_figures.py <results>/examples <output_folder>
import sys, csv, textwrap
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(f'{src}/examples.csv')))
INK, INK2, RULE = '#1d1d1b', '#5b5b57', '#c9c8c2'
GOOD, BAD = '#1a7f37', '#c62828'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})

def clean(a):
    a = a.strip()
    return a[:1].upper() + a[1:] if a else a

def make(group, ncol, fname):
    R = [r for r in rows if r['group'] == group]
    nrow = (len(R) + ncol - 1) // ncol
    pw, ih, th = 2.25, 2.25, 0.82                     # panel width, image height, text height (inches)
    fig = plt.figure(figsize=(ncol * pw + (ncol - 1) * 0.18, nrow * (ih + th) + (nrow - 1) * 0.12))
    W, H = fig.get_size_inches()
    ok = group == 'correct'
    for k, r in enumerate(R):
        i, j = divmod(k, ncol)
        x0 = (j * (pw + 0.18)) / W
        y_top = 1 - (i * (ih + th + 0.12)) / H
        axi = fig.add_axes([x0, y_top - ih / H, pw / W, ih / H])
        im = Image.open(f"{src}/{r['image_file']}").convert('RGB').resize((672, 672), Image.LANCZOS)
        axi.imshow(im); axi.set_xticks([]); axi.set_yticks([])
        for s in axi.spines.values(): s.set_edgecolor(RULE); s.set_linewidth(0.8)
        axi.text(0.03, 0.97, f'({chr(97 + k)})', transform=axi.transAxes, ha='left', va='top', fontsize=8.5,
                 fontweight='bold', color='white', bbox=dict(boxstyle='round,pad=0.25', fc='#1d1d1bcc', ec='none'))
        axt = fig.add_axes([x0, y_top - (ih + th) / H, pw / W, th / H]); axt.axis('off')
        q = textwrap.wrap(clean(r['question'].strip().rstrip(' ?')) + '?', 38)[:2]
        axt.text(0.0, 0.95, '\n'.join(q), ha='left', va='top', fontsize=7.6, color=INK, linespacing=1.25, transform=axt.transAxes)
        yb = 0.33
        axt.text(0.0, yb, 'Reference:', ha='left', va='center', fontsize=7.4, color=INK2, transform=axt.transAxes)
        axt.text(0.36, yb, clean(r['answer']), ha='left', va='center', fontsize=7.6, color=INK, fontweight='bold', transform=axt.transAxes)
        axt.text(0.0, yb - 0.25, 'Prediction:', ha='left', va='center', fontsize=7.4, color=INK2, transform=axt.transAxes)
        axt.text(0.36, yb - 0.25, clean(r['pred']) + ('  ✓' if ok else '  ✗'), ha='left', va='center', fontsize=7.6,
                 color=GOOD if ok else BAD, fontweight='bold', transform=axt.transAxes)
    fig.savefig(f'{out}/{fname}', dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.04); plt.close(fig)

make('correct', 3, 'Fig33.png'); make('incorrect', 4, 'Fig77.png'); print('ok')
