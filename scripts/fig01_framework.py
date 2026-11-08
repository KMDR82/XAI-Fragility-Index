"""Fig 1 framework diagram: draw.io XML with explicit orthogonal waypoints + a matplotlib preview of the same geometry."""
import html, sys
from pathlib import Path

G, W, BL, OR = "#f0efec", "#ffffff", "#e8f1fc", "#fdeee6"
# id: (html text, x, y, w, h, fill)
BOXES = {
    "in":   ("<b>Chest radiograph</b><br>RSNA test set, 5,337 patients<br>224 &#215; 224, grey value in 3 channels", 40, 70, 230, 70, W),
    "pert": ("<b>Perturbation</b><br>seeded per patient and condition<br>Gaussian noise &#963; = 0.1 / 0.2 / 0.3<br>Gaussian blur k = 3 / 5 / 7<br>Salt-and-pepper 1 / 2 / 3 %", 40, 175, 230, 110, W),
    "cls":  ("<b>Classifier (ResNet-50)</b><br>A: submitted model<br>A-rep: same recipe, retrained<br>A-norm: + ImageNet normalisation<br>B: ImageNet-pretrained, class-weighted", 330, 120, 230, 120, W),
    "cam":  ("<b>Explanations</b><br>for the class predicted on the clean image<br>Grad-CAM, Eigen-CAM<br>(Eigen-CAM sign-corrected: ablation)", 620, 115, 250, 80, OR),
    "pred": ("<b>Predicted class</b><br>clean vs perturbed image", 620, 330, 250, 50, W),
    "base": ("<b>Baselines</b><br>pixel random, smooth random, cross-patient,<br>randomised model (layer4 + fc), uniform", 930, 20, 280, 60, W),
    "xfi":  ("<b>XFI</b><br>RS = [SSIM + (r<sub>P</sub> + 1)/2 + (r<sub>S</sub> + 1)/2] / 3<br>XFI = 1 &#8722; RS, range [0, 4/3]", 930, 120, 280, 70, BL),
    "loc":  ("<b>Localization</b> (positive patients)<br>IoU with radiologist box, clean / perturbed<br>background activation &#8805; 0.5 / 0.7<br>energy outside the box", 930, 230, 280, 80, BL),
    "flip": ("<b>Prediction flip</b>", 930, 330, 280, 50, BL),
    "iq":   ("<b>Input degradation</b><br>PSNR, SSIM (clean vs perturbed image)", 40, 400, 230, 50, BL),
    "stat": ("<b>Statistics</b><br>patient = unit of analysis<br><br>paired Wilcoxon + Holm<br>bootstrap 95 % CI<br>linear mixed model<br>(GEE check)<br>Spearman validity", 1270, 20, 200, 430, G),
}
GROUP = ("Image pairs: clean + 9 perturbed versions", 20, 40, 270, 260)   # container behind in/pert

def c(k):  # box centre helpers
    t, x, y, w, h, _ = BOXES[k]; return x, y, w, h
def right(k, y): x, _, w, _ = c(k); return (x + w, y)
def left(k, y): x, _, _, _ = c(k); return (x, y)
def top(k, x_): _, y, _, _ = c(k); return (x_, y)
def bottom(k, x_): _, y, _, h = c(k); return (x_, y + h)

gx, gy, gw, gh = GROUP[1:]
# edges: list of polylines (absolute points); label optional
EDGES = [
    ("in", "pert", [bottom("in", 155), top("pert", 155)], ""),
    ("grp", "cls", [(gx + gw, 180), left("cls", 180)], ""),
    ("grp", "iq", [(155, gy + gh), top("iq", 155)], ""),
    ("cls", "cam", [right("cls", 155), left("cam", 155)], ""),
    ("cls", "pred", [bottom("cls", 445), (445, 355), left("pred", 355)], ""),
    ("cam", "xfi", [right("cam", 155), left("xfi", 155)], ""),
    ("cam", "loc", [bottom("cam", 800), (800, 270), left("loc", 270)], ""),
    ("base", "xfi", [bottom("base", 1070), top("xfi", 1070)], "reference"),
    ("pred", "flip", [right("pred", 355), left("flip", 355)], ""),
    ("xfi", "stat", [right("xfi", 155), left("stat", 155)], ""),
    ("loc", "stat", [right("loc", 270), left("stat", 270)], ""),
    ("flip", "stat", [right("flip", 355), left("stat", 355)], ""),
    ("iq", "stat", [right("iq", 425), left("stat", 425)], ""),
]

def drawio(path):
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>',
             f'<mxCell id="grp" value="{html.escape(GROUP[0])}" style="rounded=1;whiteSpace=wrap;html=1;fillColor={G};strokeColor=#9a9893;dashed=1;'
             f'verticalAlign=top;fontSize=10;fontColor=#52514e;arcSize=4;" vertex="1" parent="1"><mxGeometry x="{gx}" y="{gy}" width="{gw}" height="{gh}" as="geometry"/></mxCell>']
    for k, (txt, x, y, w, h, fill) in BOXES.items():
        cells.append(f'<mxCell id="{k}" value="{html.escape(txt)}" style="rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor=#52514e;fontSize=11;'
                     f'fontFamily=Helvetica;arcSize=8;spacing=4;" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for i, (a, b, pts, lab) in enumerate(EDGES):
        (sx, sy), (tx, ty) = pts[0], pts[-1]
        mid = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in pts[1:-1])
        cells.append(f'<mxCell id="e{i}" value="{lab}" style="edgeStyle=none;rounded=0;html=1;endArrow=block;endFill=1;strokeColor=#52514e;strokeWidth=1.3;'
                     f'fontSize=10;labelBackgroundColor=#ffffff;" edge="1" parent="1" source="{a}" target="{b}"><mxGeometry relative="1" as="geometry">'
                     f'<mxPoint x="{sx}" y="{sy}" as="sourcePoint"/><mxPoint x="{tx}" y="{ty}" as="targetPoint"/>'
                     + (f'<Array as="points">{mid}</Array>' if mid else "") + '</mxGeometry></mxCell>')
    xml = ('<mxfile host="app.diagrams.net"><diagram name="Fig01" id="fig01"><mxGraphModel dx="1500" dy="600" grid="1" gridSize="10" page="1" '
           'pageWidth="1500" pageHeight="480"><root>' + "".join(cells) + '</root></mxGraphModel></diagram></mxfile>')
    Path(path).write_text(xml, encoding="utf-8")

def preview(path):
    import re, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(15, 4.9)); ax.set_xlim(0, 1490); ax.set_ylim(470, 0); ax.axis("off")
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh, "round,pad=0,rounding_size=6", fc=G, ec="#9a9893", ls="--"))
    ax.text(gx + gw / 2, gy + 10, GROUP[0], ha="center", va="top", fontsize=7, color="#52514e")
    def plain(t):
        t = t.replace("<br>", "\n").replace("<sub>P</sub>", "_P").replace("<sub>S</sub>", "_S")
        return html.unescape(re.sub("<[^>]+>", "", t))
    for k, (txt, x, y, w, h, fill) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, "round,pad=0,rounding_size=6", fc=fill, ec="#52514e"))
        ax.text(x + w / 2, y + h / 2, plain(txt), ha="center", va="center", fontsize=6.6, linespacing=1.25)
    for a, b, pts, lab in EDGES:
        xs, ys = zip(*pts); ax.plot(xs[:-1] + (xs[-1],), ys, color="#52514e", lw=1.1)
        ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=1.1, shrinkA=0, shrinkB=0))
        if lab: ax.text(xs[0] + 4, (ys[0] + ys[-1]) / 2, lab, fontsize=6.5, va="center", backgroundcolor="white")
    fig.savefig(path, dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    drawio(out / "Fig01_framework.drawio"); preview(out / "Fig01_framework_preview.png"); print("ok")
