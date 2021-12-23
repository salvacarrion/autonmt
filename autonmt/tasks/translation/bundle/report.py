import os
import pandas as pd
from autonmt.utils import make_dir, save_json
from autonmt import plots


def generate_report(scores, metric_id, output_path, save_figures=True, show_figures=False):
    # Create logs path
    scores_path = os.path.join(output_path, "scores")
    plots_path = os.path.join(output_path, "plots")
    make_dir([scores_path, plots_path])

    # Save scores
    _ = save_scores_as_json(output_path=scores_path, scores=scores)
    df_metrics = save_scores_as_pandas(output_path=scores_path, scores=scores)

    # Plot metrics
    plots.plot_metrics(output_path=plots_path, df_metrics=df_metrics, metric_id=metric_id, save_figures=save_figures,
                       show_figures=show_figures)


def save_scores_as_json(output_path, scores):
    # Save json metrics
    json_metrics_path = os.path.join(output_path, "scores.json")
    save_json(scores, json_metrics_path)
    return scores


def save_scores_as_pandas(output_path, scores):
    # Convert to pandas
    rows = []
    for model_scores in scores:
        for eval_scores in model_scores:
            eval_scores = dict(eval_scores)  # Copy
            beams_unrolled = {f"{beam_width}__{k}": v for beam_width in eval_scores["beams"].keys() for k, v in
                              eval_scores["beams"][beam_width].items()}
            eval_scores.pop("beams")
            eval_scores.update(beams_unrolled)
            rows.append(eval_scores)

    # Convert to pandas
    df = pd.DataFrame(rows)
    csv_scores_path = os.path.join(output_path, "scores.csv")
    df.to_csv(csv_scores_path, index=False)
    return df