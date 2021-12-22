import pandas as pd
from autonmt.utils import *
from autonmt import plots


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


def create_report(scores, metric_id, output_path, save_figures=True, show_figures=False):
    # Create logs path
    scores_path = os.path.join(output_path, "scores")
    plots_path = os.path.join(output_path, "plots")
    make_dir([scores_path, plots_path])

    # Save scores
    df_metrics = save_scores(output_path=scores_path, scores=scores)

    # Plot metrics
    plots.plot_metrics(output_path=plots_path, df_metrics=df_metrics, metric_id=metric_id, save_figures=save_figures,
                       show_figures=show_figures)


def save_scores(output_path, scores):
    # Save json metrics
    json_metrics_path = os.path.join(output_path, "scores.json")
    save_json(scores, json_metrics_path)

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


