import os
import pandas as pd
from autonmt.utils import make_dir, save_json
from autonmt import plots


def generate_report(scores, output_path, plot_metric=None, **kwargs):
    # Create logs path
    reports_path = os.path.join(output_path, "reports")
    plots_path = os.path.join(output_path, "plots")
    make_dir([reports_path, plots_path])

    # Convert scores to pandas
    df_report = scores2pandas(scores=scores)

    # Create summary
    df_summary = summarize_scores([df_report])

    # Save report: json
    json_report_path = os.path.join(reports_path, "report.json")
    save_json(scores, json_report_path)

    # Save report: pandas
    csv_report_path = os.path.join(reports_path, "report.csv")
    df_report.to_csv(csv_report_path, index=False)

    # Save report: pandas
    csv_summary_path = os.path.join(reports_path, "report_summary.csv")
    df_summary.to_csv(csv_summary_path, index=False)

    # Plot metrics
    if plot_metric:
        plots.plot_metrics(output_path=plots_path, df_report=df_report.copy(), plot_metric=plot_metric, **kwargs)

    return df_report, df_summary


def scores2pandas(scores):
    # Convert to pandas
    rows = []
    for model_scores in scores:
        for eval_scores in model_scores:
            eval_scores = dict(eval_scores)  # Copy
            beams_unrolled = {f"{beam_width}__{m_name_full}": score for beam_width in eval_scores["beams"].keys() for m_name_full, score in eval_scores["beams"].get(beam_width, {}).items()}
            config_fit_unrolled = {f"config_fit_{key}":  str(value) for key, value in eval_scores["config"].get("fit", {}).items()}
            config_predict_unrolled = {f"config_predict_{key}": str(value) for key, value in eval_scores["config"].get("predict", {}).items()}
            eval_scores.update(beams_unrolled)
            eval_scores.update(config_fit_unrolled)
            eval_scores.update(config_predict_unrolled)
            eval_scores.pop("beams")
            eval_scores.pop("config")
            rows.append(eval_scores)

    # Convert to pandas
    df = pd.DataFrame(rows)
    return df


def summarize_scores(scores_collection, cols=None, ref_metric="bleu"):
    if cols is None:
        cols = ["train_dataset", "eval_dataset", "subword_model", "vocab_size"]

    collections = []
    for c in scores_collection:
        collections.append([row for i, row in c.iterrows()])

    rows = []
    for run_scores in zip(*collections):
        # This MUST be fixed to compare several toolkits
        row = {}
        for c in cols:
            row[c] = run_scores[0][c]

        # Add scores from other toolkits (including itself)
        for m_scores in run_scores:
            m_col = [x for x in list(m_scores.keys()) if '_' + ref_metric.lower() in x.lower()]
            m_col.sort(reverse=True)  # Descending (beam 5 before 1)
            row[f"{m_scores['engine']}_{ref_metric}"] = m_scores[m_col[0]]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df
