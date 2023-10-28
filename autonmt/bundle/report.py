import os

import pandas as pd

from autonmt.bundle import plots
from autonmt.bundle.utils import make_dir, save_json


def generate_report(scores, output_path, plot_metric=None, **kwargs):
    if not scores:
        raise ValueError("No scores were given")

    # Create logs path
    reports_path = os.path.join(output_path, "reports")
    plots_path = os.path.join(output_path, "plots")
    make_dir([reports_path, plots_path])

    # Convert scores to pandas
    df_report = scores2pandas(scores=scores)

    # Create summary
    df_summary = summarize_scores(df_report)

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
        plots.plot_metrics(output_path=plots_path, df_report=df_report, plot_metric=plot_metric, **kwargs)

    return df_report, df_summary


def scores2pandas(scores):
    # Convert to pandas
    pd_rows = []
    for model_scores in scores:
        for eval_scores in model_scores:
            pd_rows.append(pd.json_normalize(eval_scores))

    if not pd_rows:
        raise ValueError("=> [Report]: No scores were given")

    # Convert to pandas
    df = pd.concat(pd_rows)
    return df


def summarize_scores(df_report, default_cols=None, ref_metric="bleu"):
    if default_cols is None:
        default_cols = ["train_dataset", "eval_dataset", "lang_pair", "subword_model", "vocab_size"]

    # Select columns
    selected_cols = [c for c in df_report.columns.values if c in default_cols or ref_metric in c]
    df = df_report[selected_cols]
    return df


def generate_multivariable_report(data, output_path, x, y_left, y_right=None, loc_legend="upper left",
                                  prefix="", save_csv=False, **kwargs):
    # Create logs path
    reports_path = os.path.join(output_path, "reports")
    plots_path = os.path.join(output_path, "plots")
    make_dir([reports_path, plots_path])

    # Save dataframes
    if save_csv:
        data.to_csv(os.path.join(reports_path, f"{prefix}_vocabs_report.csv"), index=False)

    # Plot vocabs report
    plots.plot_vocabs_report(output_path=plots_path, data=data, x=x, y_left=y_left, y_right=y_right,
                             loc_legend=loc_legend, prefix=prefix, **kwargs)

