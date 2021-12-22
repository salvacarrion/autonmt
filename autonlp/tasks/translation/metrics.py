import os
import json
import re

import pandas as pd
from autonlp.utils import *
from autonlp import plots


def parse_sacrebleu(text):
    result = {}
    metrics = json.loads("".join(text))
    metrics = [metrics] if isinstance(metrics, dict) else metrics

    for m_dict in metrics:
        m_name = m_dict['name'].lower().strip()
        result[m_name] = float(m_dict["score"])
    return result


def parse_bertscore(text):
    pattern = r"P: ([01]\.\d*) R: ([01]\.\d*) F1: ([01]\.\d*)"
    line = text[-1].strip()
    groups = re.search(pattern, line).groups()
    result = {"precision": float(groups[0]), "recall": float(groups[1]), "f1": float(groups[2])}
    return result


def parse_comet(text):
    pattern = r"score: (-?[01]\.\d*)"
    line = text[-1].strip()
    groups = re.search(pattern, line).groups()
    result = {"score": float(groups[0])}
    return result


def parse_beer(text):
    pattern = r"total BEER ([01]\.\d*)"
    line = text[-1].strip()
    groups = re.search(pattern, line).groups()
    result = {"score": float(groups[0])}
    return result


def create_report(metrics, metric_id, output_path, save_figures=True, show_figures=False):
    # Create logs path
    metrics_path = os.path.join(output_path, "metrics")
    plots_path = os.path.join(output_path, "plots")
    make_dir([metrics_path, plots_path])

    # Save scores
    df_metrics = save_metrics(output_path=metrics_path, metrics=metrics)

    # Plot metrics
    plots.plot_metrics(output_path=plots_path, df_metrics=df_metrics, metric_id=metric_id, save_figures=save_figures,
                       show_figures=show_figures)


def save_metrics(output_path, metrics):
    # Save json metrics
    json_metrics_path = os.path.join(output_path, "metrics.json")
    save_json(metrics, json_metrics_path)

    # Convert to pandas
    rows = []
    for ds_train_name, ds_train_evals in metrics.items():
        for ds_eval_name, ds_eval_scores in ds_train_evals.items():
            scores = dict(ds_eval_scores)  # Copy
            beams_unrolled = {f"{beam_width}__{k}": v for beam_width in ds_eval_scores["beams"].keys() for k, v in
                              scores["beams"][beam_width].items()}
            scores.pop("beams")
            scores.update(beams_unrolled)
            rows.append(scores)

    # Convert to pandas
    df = pd.DataFrame(rows)
    csv_metrics_path = os.path.join(output_path, "metrics.csv")
    df.to_csv(csv_metrics_path, index=False)
    return df


