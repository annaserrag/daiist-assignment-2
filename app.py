"""
Assignment 2 — Gradio dashboard for the seal check.

Run with `uv run python main.py app`. It loads the committed weights in weights/ and
the small tables train.py saved in artifacts/, and never trains anything: every number
on the page is recomputed from the stored page scores of the test split, so moving a
slider re-costs the real pipeline output rather than a copy of it.

    Inspect a page      upload a page, see the boxes and the verdict at the current band
    Operating point     the two thresholds -> confusion matrix, cost per 1,000 pages,
                        share sent to an associate, people needed
    Cost curve          cost against a single threshold, chosen operating point marked
    Per-class results   precision, recall, AP per class, the confusion matrix, baselines
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import costing

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
EXAMPLES = ROOT / "examples"

SUMMARY = json.loads((ARTIFACTS / "summary.json").read_text())
MODEL = YOLO(str(ROOT / SUMMARY["weights"]))
PER_CLASS = pd.read_csv(ARTIFACTS / "per_class.csv")
BASELINES = pd.read_csv(ARTIFACTS / "baselines.csv")
COMPOSITION = pd.read_csv(ARTIFACTS / "composition.csv")
SENSITIVITY = pd.read_csv(ARTIFACTS / "sensitivity.csv")
RELIABILITY = pd.read_csv(ARTIFACTS / "reliability.csv")
SWEEP_TEST = pd.read_csv(ARTIFACTS / "sweep_test.csv")
SWEEP_VAL = pd.read_csv(ARTIFACTS / "sweep_val.csv")
PAGES = pd.read_csv(ARTIFACTS / "page_scores.csv")

# The test pages the decision is costed on: real document pages plus the stress-test
# pages with the seal painted out (train.py, 4.2). QR photographs are not data room pages.
TEST_PAGES = PAGES[(PAGES.split == "test") & (PAGES.source != "qr photo")]
UNSEALED = TEST_PAGES.loc[~TEST_PAGES.sealed, "score"].to_numpy()
SEALED = TEST_PAGES.loc[TEST_PAGES.sealed, "score"].to_numpy()

LOW0, HIGH0 = SUMMARY["chosen_band"]["low"], SUMMARY["chosen_band"]["high"]
CHOSEN_CUT = SUMMARY["chosen_cut"]
BOX_COLOURS = {"stamp": "#16a34a", "Signature": "#dc2626", "Text": "#2563eb",
               "qr-code": "#d97706"}
VERDICT = {"pass": "PASS: sealed, nobody looks at it",
           "review": "REVIEW: an associate checks this page",
           "fail": "FAIL: flagged as unsealed, executed copy requested"}


def zone(score, low, high):
    if score >= high:
        return "pass"
    return "fail" if score < low else "review"


def clean_band(low, high):
    """Keep the band valid when a slider is dragged past the other one."""
    return min(low, high), max(low, high)


# --------------------------------------------------------------------------------
# Inspect a page
# --------------------------------------------------------------------------------
def inspect_page(image, low, high):
    if image is None:
        return None, "Upload a page, or pick one of the examples."
    low, high = clean_band(low, high)
    image = image.convert("RGB")
    result = MODEL.predict(image, conf=0.001, verbose=False)[0]
    boxes = list(zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                     result.boxes.xyxy.tolist()))
    score = costing.page_score([(c, s) for c, s, _ in boxes], MODEL.names)

    # Draw what the model reports at the current threshold: every box at or above the
    # fail cut, since anything below it is treated as not there.
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=max(14, image.size[1] // 45))
    shown = [(c, s, b) for c, s, b in boxes if s >= max(low, 0.05)]
    for c, s, (x0, y0, x1, y1) in shown:
        name = MODEL.names[int(c)]
        colour = BOX_COLOURS.get(name, "black")
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=4)
        draw.text((x0 + 3, max(0, y0 - font.size - 4)), f"{name} {s:.2f}",
                  fill=colour, font=font)

    verdict = zone(score, low, high)
    lines = [f"### {VERDICT[verdict]}",
             f"Stamp score (most confident stamp box): **{score:.3f}**  ",
             f"Band: fail below {low:.3f}, pass at or above {high:.3f}", ""]
    lines += [f"- `{MODEL.names[int(c)]}` {s:.3f}" for c, s, _ in shown] or [
        "- no boxes above the fail cut"]
    return annotated, "\n".join(lines)


# --------------------------------------------------------------------------------
# Operating point
# --------------------------------------------------------------------------------
def operating_point(low, high, base_rate, pages_per_day):
    low, high = clean_band(low, high)
    row = costing.three_way_cost(UNSEALED, SEALED, low, high, base_rate=base_rate)

    # Raw test pages in each zone. Counts, so the reader sees how thin each cell is.
    raw = pd.DataFrame(
        [[int(np.sum([zone(s, low, high) == z for s in scores])) for z in
          ("fail", "review", "pass")] for scores in (UNSEALED, SEALED)],
        index=["truth: unsealed (positive)", "truth: sealed"],
        columns=["model: fail", "model: review", "model: pass"],
    ).reset_index(names="test pages")

    # Final outcome per 1,000 data room pages, after the associate has seen the band.
    projected = pd.DataFrame(
        [["flagged as unsealed", row["tp"], row["fp"]],
         ["passed as sealed", row["fn"], row["tn"]]],
        columns=["per 1,000 pages", "truth: unsealed", "truth: sealed"],
    ).round(2)

    designs = costing.reference_designs(UNSEALED, SEALED, base_rate=base_rate)
    designs["this band"] = row
    alt = pd.DataFrame([{"design": k, "EUR per 1,000 pages": round(v["cost"], 0),
                         "missed seals": round(v["fn"], 2),
                         "sent to associate": f"{v['review_share']:.1%}"}
                        for k, v in designs.items()])

    staff = costing.staffing(row["review_share"], pages_per_day, row["flagged"])
    text = (
        f"### EUR {row['cost']:,.0f} per 1,000 pages\n"
        f"- **{row['review_share']:.1%}** of pages sent to an associate "
        f"({staff['reviewed_per_day']:.0f} a day at {pages_per_day} pages a day)\n"
        f"- {staff['flagged_per_day']:.0f} pages a day flagged and raised with the seller\n"
        f"- {staff['hours_per_day']:.1f} associate hours a day, "
        f"**{staff['associates']:.2f} associates** at "
        f"{costing.REVIEW_HOURS_PER_DAY} productive hours each\n"
        f"- missed unsealed pages: {row['fn']:.2f} per 1,000 "
        f"(base rate {base_rate:.0%}: {1000 * base_rate:.0f} unsealed per 1,000)"
    )
    return raw, projected, text, alt


# --------------------------------------------------------------------------------
# Cost curve
# --------------------------------------------------------------------------------
def cost_curve(low, high, base_rate):
    low, high = clean_band(low, high)
    test = pd.DataFrame(costing.single_cut_sweep(UNSEALED, SEALED, base_rate=base_rate))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["low"], y=test["cost"], mode="lines",
                             name="single cut, test pages", line={"width": 3}))
    if np.isclose(base_rate, costing.BASE_RATE):
        fig.add_trace(go.Scatter(x=SWEEP_VAL["low"], y=SWEEP_VAL["cost"], mode="lines",
                                 name="single cut, validation pages",
                                 line={"dash": "dot"}))
    chosen = test.loc[np.isclose(test["low"], CHOSEN_CUT)]
    if not chosen.empty:
        fig.add_trace(go.Scatter(
            x=[CHOSEN_CUT], y=chosen["cost"], mode="markers+text",
            marker={"size": 13, "color": "#dc2626"}, name="chosen cut (from validation)",
            text=[f"chosen {CHOSEN_CUT:.3f}"], textposition="top center"))
    band = costing.three_way_cost(UNSEALED, SEALED, low, high, base_rate=base_rate)
    fig.add_vrect(x0=low, x1=max(high, low + 0.002), fillcolor="#f59e0b", opacity=0.15,
                  line_width=0, annotation_text=f"review band: EUR {band['cost']:,.0f}",
                  annotation_position="top left")
    fig.add_vline(x=SUMMARY["t_star_if_calibrated"], line_dash="dash", line_color="grey",
                  annotation_text="t* if calibrated", annotation_position="bottom right")
    fig.update_layout(
        xaxis_title="threshold on the stamp score (fail below, pass at or above)",
        yaxis_title="EUR per 1,000 pages", yaxis_type="log", height=480,
        legend={"orientation": "h", "y": -0.25}, margin={"t": 30},
    )
    return fig


# --------------------------------------------------------------------------------
# Per-class results
# --------------------------------------------------------------------------------
def confusion_figure():
    cm = SUMMARY["confusion_matrix"]
    matrix = np.array(cm["matrix"])
    fig = go.Figure(go.Heatmap(z=matrix, x=cm["labels"], y=cm["labels"], text=matrix,
                               texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(xaxis_title="true class", yaxis_title="predicted class",
                      yaxis_autorange="reversed", height=420, margin={"t": 20})
    return fig


def per_class_table():
    table = PER_CLASS.copy()
    table["trust"] = np.where(table["test_instances"] < 100,
                              "too few test instances to trust", "")
    return table.round(3)


# --------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------
with gr.Blocks(title="Seal check") as demo:
    gr.Markdown(
        "# Seal check for due diligence\n"
        "Every execution page in the target's data room must carry the company seal. "
        "The finetuned YOLO11s scores each page by its most confident **stamp** box. "
        "Positive class: an **unsealed** page. Below the fail cut the page is flagged, "
        "at or above the pass cut it is passed, in between an associate looks at it.  \n"
        f"Chosen on validation: fail below **{LOW0:.3f}**, pass at or above "
        f"**{HIGH0:.3f}**. Overall test mAP50 {SUMMARY['overall']['mAP50']:.3f}, "
        f"mAP50-95 {SUMMARY['overall']['mAP50-95']:.3f}."
    )
    with gr.Row():
        low = gr.Slider(0, 1, value=LOW0, step=0.0025, label="Fail below (low cut)")
        high = gr.Slider(0, 1.01, value=HIGH0, step=0.0025, label="Pass at or above (high cut)")
        base_rate = gr.Slider(0.005, 0.30, value=costing.BASE_RATE, step=0.005,
                              label="Base rate: share of pages with no seal")
        pages_per_day = gr.Number(value=costing.PAGES_PER_DAY, precision=0,
                                  label="Pages a day")

    with gr.Tab("Inspect a page"):
        with gr.Row():
            page_in = gr.Image(type="pil", label="Execution page", sources=["upload", "clipboard"])
            page_out = gr.Image(label="Detections at the current threshold")
        verdict_md = gr.Markdown()
        examples = sorted(str(p) for p in EXAMPLES.glob("*.jpg"))
        if examples:
            gr.Examples(examples, inputs=page_in)

    with gr.Tab("Operating point"):
        op_text = gr.Markdown()
        with gr.Row():
            raw_table = gr.Dataframe(label="Test pages by zone (counts)")
            projected_table = gr.Dataframe(label="Outcome per 1,000 data room pages, "
                                                 "after the associate")
        alt_table = gr.Dataframe(label="Against the alternatives, at this base rate")

    with gr.Tab("Cost curve"):
        curve = gr.Plot()
        gr.Markdown(
            "The curve is one cut and no review. The shaded band is the current pair of "
            "cuts and its cost with an associate on the band. The cost axis is "
            "logarithmic because doing nothing is two orders of magnitude dearer than "
            "any sensible cut."
        )
        gr.Dataframe(SENSITIVITY.round(4), label="Which assumption moves the band")
        gr.Dataframe(RELIABILITY.round(3),
                     label="Is the stamp score a probability? Share actually sealed per "
                           "score bin, test pages")

    with gr.Tab("Per-class results"):
        gr.Dataframe(per_class_table(), label="Finetuned YOLO11s, test split, per class")
        gr.Plot(confusion_figure())
        gr.Markdown("Rows are what the model predicted, columns the labelled class, "
                    "matched at IoU 0.45, boxes at confidence 0.25 or above. "
                    "'background' is a missed object (column) or a box on nothing (row).")
        gr.Dataframe(BASELINES.round(3), label="Stamp class on test: finetuned vs baselines")
        gr.Dataframe(COMPOSITION[COMPOSITION.split == "test"].round(4),
                     label="Test split composition")

    page_in.change(inspect_page, [page_in, low, high], [page_out, verdict_md])
    for control in (low, high, base_rate, pages_per_day):
        control.change(operating_point, [low, high, base_rate, pages_per_day],
                       [raw_table, projected_table, op_text, alt_table])
    for control in (low, high, base_rate):
        control.change(cost_curve, [low, high, base_rate], curve)
    for control in (low, high):
        control.change(inspect_page, [page_in, low, high], [page_out, verdict_md])
    demo.load(operating_point, [low, high, base_rate, pages_per_day],
              [raw_table, projected_table, op_text, alt_table])
    demo.load(cost_curve, [low, high, base_rate], curve)


if __name__ == "__main__":
    demo.launch()
