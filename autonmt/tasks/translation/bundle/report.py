import os
import pandas as pd
from autonmt.utils import make_dir, save_json
from autonmt import plots


def generate_report(scores, metric_id, output_path, save_figures=True, show_figures=False):
    # Create logs path
    scores_path = os.path.join(output_path, "scores")
    plots_path = os.path.join(output_path, "plots")
    make_dir([scores_path, plots_path])

    # Convert scores to pandas
    df_metrics = scores2pandas(scores=scores)

    # Save scores: json
    _ = save_scores_as_json(output_path=scores_path, scores=scores)

    # Save scores: pandas
    csv_scores_path = os.path.join(output_path, "scores.csv")
    df_metrics.to_csv(csv_scores_path, index=False)

    # Plot metrics
    plots.plot_metrics(output_path=plots_path, df_metrics=df_metrics, metric_id=metric_id, save_figures=save_figures,
                       show_figures=show_figures)


def save_scores_as_json(output_path, scores):
    # Save json metrics
    json_metrics_path = os.path.join(output_path, "scores.json")
    save_json(scores, json_metrics_path)
    return scores


def scores2pandas(scores):
    # Convert to pandas
    rows = []
    for model_scores in scores:
        for eval_scores in model_scores:
            eval_scores = dict(eval_scores)  # Copy
            beams_unrolled = {f"{beam_width}__{m_name_full}": score for beam_width in eval_scores["beams"].keys() for m_name_full, score in eval_scores["beams"][beam_width].items()}
            eval_scores.pop("beams")
            eval_scores.update(beams_unrolled)
            rows.append(eval_scores)

    # Convert to pandas
    df = pd.DataFrame(rows)
    return df


def summarize_scores(scores_collection, beam_width=1):
    collections = []
    for c in scores_collection:
        collections.append([row for i, row in c.iterrows()])

    rows = []
    for run_scores in zip(*collections):
        row = {"subword_model": run_scores[0]["subword_model"], "vocab_size": run_scores[0]["vocab_size"]}
        for m_scores in run_scores:
            row[f"{m_scores['engine']}_bleu"] = m_scores[f"beam{beam_width}__sacrebleu_bleu_score"]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df
