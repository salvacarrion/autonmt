import os
import pandas as pd
from autonmt.bundle.utils import make_dir
from autonmt.bundle.plots import plot_metrics

BASE_PATH = "/Users/salvacarrion/Documents/Programming/datasets/translate"  # Local


def bar_group_name_fn(df_row):
    # Set default values for the columns
    return f"{df_row['train_dataset'].replace('_', ' ')}".title()

def legend_name_fn(text):
    # Set default values for the columns
    return text.title().replace('_', ' ')

def main():
    # Read report
    report_path = "/Users/salvacarrion/Documents/Programming/datasets/translate/.outputs/autonmt/2023-10-28 02:10:46.853218/reports/report_summary2.csv"
    df_report = pd.read_csv(report_path)
    print(df_report.to_string(index=False))

    # Plot metrics
    plots_path = os.path.join(BASE_PATH, "plots")
    make_dir([plots_path])

    #Create plots path
    plot_metrics(output_path=plots_path, df_report=df_report, plot_metric="translations.beam1.sacrebleu_bleu_score",
                 xlabel="MT Models", ylabel="BLEU Score", title="Model comparison",
                 bar_group_name_fn=bar_group_name_fn, legend_name_fn=legend_name_fn)
    print("Plots done!")


if __name__ == "__main__":
    main()
