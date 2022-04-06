import os
from pathlib import Path

# Sort of incompatible with plotly
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from autonmt.bundle import utils

sns.set()


def set_non_gui_backend():
    # If this is not set, matplotlib won't most of the images that were created a loop
    # Source: https://github.com/matplotlib/matplotlib/issues/8519
    matplotlib.use('agg')


def catplot(data, x, y, hue, title, xlabel, ylabel, leyend_title, output_dir, fname, aspect_ratio=(12, 8), size=1.0,
            show_values=True, dpi=150, rotate_xlabels=0, show_fig=False, save_fig=True, formats=None,
            overwrite=True, data_format='{:.0f}', loc="upper right"):
    if formats is None:
        formats = ["png", "pdf"]

    # Check if the figures exists
    if not overwrite and (save_fig and do_all_figs_exists(output_dir, fname, formats)):
        print(f"\t\t\t- Skipped catplot as it already exists ({fname})")
        return False

    # Create subplot
    fig = plt.figure(figsize=(aspect_ratio[0], aspect_ratio[1]))
    sns.set(font_scale=size)

    # Plot catplot
    g = sns.catplot(data=data, x=x, y=y, hue=hue, kind="bar", legend=False, height=aspect_ratio[1],
                    aspect=aspect_ratio[0] / aspect_ratio[1])

    # Tweaks
    g.set(xlabel=xlabel, ylabel=ylabel)
    g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=rotate_xlabels)
    # g.ax.tick_params(axis='x', which='major', labelsize=8*size)
    # g.ax.tick_params(axis='y', which='major', labelsize=8*size)
    g.ax.yaxis.set_major_formatter(utils.human_format_int)  # only for catplot

    # Add values
    if show_values:
        ax = g.facet_axis(0, 0)
        for c in ax.containers:
            labels = [data_format.format(float(v.get_height())) for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')  #, fontsize=8 * size

    # properties
    g.set(xlabel=xlabel, ylabel=ylabel)
    plt.title(title)
    plt.legend(title=leyend_title, loc=loc) if hue else None
    plt.tight_layout()

    # Show/Save/Close figure
    _show_save_figure(output_dir, fname, show_fig, save_fig, formats, dpi, fig)


def barplot(data, x, y, output_dir, fname, title="", xlabel="x", ylabel="y", aspect_ratio=(12, 8), size=1.0,
            dpi=150, show_fig=False, save_fig=True, formats=None, overwrite=True):
    if formats is None:
        formats = ["png", "pdf"]

    # Check if the figures exists
    if not overwrite and (save_fig and do_all_figs_exists(output_dir, fname, formats)):
        print(f"\t\t\t- Skipped barplot as it already exists ({fname})")
        return False

    # Create subplot
    fig = plt.figure(figsize=(aspect_ratio[0] * size, aspect_ratio[1] * size))
    sns.set(font_scale=size)

    # Plot barplot
    g = sns.barplot(data=data, x=x, y=y)

    # Tweaks
    g.set(xlabel=xlabel, ylabel=ylabel)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.tick_params(axis='x', which='major', labelsize=8)  # *size  => because of the vocabulary distribution
    g.tick_params(axis='y', which='major', labelsize=8)  # *size  => because of the vocabulary distribution
    g.yaxis.set_major_formatter(utils.human_format_int)

    # properties
    plt.title(title)
    plt.tight_layout()

    # Show/Save/Close figure
    _show_save_figure(output_dir, fname, show_fig, save_fig, formats, dpi, fig)


def histogram(data, x, output_dir, fname, title="", xlabel="x", ylabel="y", bins="auto", aspect_ratio=(12, 8), size=1.0,
              dpi=150, show_fig=False, save_fig=True, formats=None, overwrite=True):
    if formats is None:
        formats = ["png", "pdf"]

    # Check if the figures exists
    if not overwrite and (save_fig and do_all_figs_exists(output_dir, fname, formats)):
        return False

    # Create subplot
    fig = plt.figure(figsize=(aspect_ratio[0] * size, aspect_ratio[1] * size))
    sns.set(font_scale=size)

    # Plot barplot
    g = sns.histplot(data=data, x=x, bins=bins)

    # Tweaks
    g.set(xlabel=xlabel, ylabel=ylabel)
    # g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.tick_params(axis='x', which='major', labelsize=8 * size)
    g.tick_params(axis='y', which='major', labelsize=8 * size)
    g.yaxis.set_major_formatter(utils.human_format_int)

    # properties
    plt.title(title)
    plt.tight_layout()

    # Show/Save/Close figure
    _show_save_figure(output_dir, fname, show_fig, save_fig, formats, dpi, fig)


def heatmap(data, xlabels, ylabels, output_dir, fname, title="", annot=True, cbar=False, aspect_ratio=(12, 12),
            font_scale=1.0, dpi=150, show_fig=False, save_fig=True, formats=None, overwrite=True, annot_format=".2f"):
    if formats is None:
        formats = ["png", "pdf"]

    # Check if the figures exists
    if not overwrite and (save_fig and do_all_figs_exists(output_dir, fname, formats)):
        return False

    # Create subplot
    fig = plt.figure(figsize=(aspect_ratio[0], aspect_ratio[1]))
    sns.set(font_scale=font_scale)

    # Plot barplot
    g = sns.heatmap(data, annot=annot, cbar=cbar, fmt=annot_format)
    g.set_xticklabels([x.title() for x in xlabels], ha='center', minor=False)
    g.set_yticklabels([y.title() for y in ylabels], va='center', minor=False)

    # Tweaks
    # g.set_xticklabels(g.get_xticklabels(), rotation=90)
    # g.tick_params(axis='x', which='major', labelsize=8 * size)
    # g.tick_params(axis='y', which='major', labelsize=8 * size)
    # g.yaxis.set_major_formatter(utils.human_format_int)

    # properties
    plt.title(title, y=1.01) if title else None
    plt.tight_layout()

    # Show/Save/Close figure
    _show_save_figure(output_dir, fname, show_fig, save_fig, formats, dpi, fig)


def lineplot(data, x, y_left, y_left_hue, title, xlabel, ylabel_left, leyend_title, output_dir,
            fname, y_right=None, y_right_hue=None, ylabel_right=None, aspect_ratio=(12, 8), size=1.0, show_values=True, dpi=300,
             rotate_xlabels=0, show_fig=False, save_fig=True, formats=None, overwrite=True, data_format='{:.0f}',
             loc_legend="upper left"):
    if formats is None:
        formats = ["png", "pdf"]

    # Check if the figures exists
    if not overwrite and (save_fig and do_all_figs_exists(output_dir, fname, formats)):
        print(f"\t\t\t- Skipped lineplot as it already exists ({fname})")
        return False

    # Create subplot
    fig, ax = plt.subplots(1, 1, figsize=(aspect_ratio[0] * size, aspect_ratio[1] * size))  # figsize(W, H)

    sns.set(font_scale=1.5)

    # Plot lines
    g1 = sns.lineplot(data=data, x=x, y=y_left, hue=y_left_hue, ax=ax, marker="o", legend=True)
    g1.set(ylim=(0, None))
    g1.set(xlabel=xlabel, ylabel=ylabel_left)
    h1, l1 = g1.get_legend_handles_labels()

    # Set secondary plot
    _g, _ax = g1, ax
    h2, l2 = [], []
    if y_right:
        ax2 = plt.twinx()
        ax2.grid(False)
        g2 = sns.lineplot(data=data, x=x, y=y_right, ax=ax2, color="grey", linestyle="dashed", label=ylabel_right, legend=False)
        g2.set(xlabel=xlabel, ylabel=ylabel_right)
        h2, l2 = g2.get_legend_handles_labels()
        _g, _ax = g2, ax2
        ax.get_legend().remove()

    # Set legend
    _g.legend(loc=loc_legend, handles=h1 + h2, labels=l1 + l2)

    # properties
    plt.title(title)
    plt.tight_layout()

    # Show/Save/Close figure
    _show_save_figure(output_dir, fname, show_fig, save_fig, formats, dpi, fig)
    asd = 33


def _show_save_figure(output_dir, fname, show_fig, save_fig, formats, dpi, fig=None):
    # Save image
    if save_fig:
        for ext in formats:
            # Create png/pdf/... dirs
            save_dir = os.path.join(output_dir, ext)
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            # Save image
            filename = os.path.join(save_dir, f"{fname}.{ext}")
            plt.savefig(filename, dpi=dpi)
            print(f"\t\t\t- Figure saved: {filename}")

    # Show plot
    if show_fig:
        plt.show()
        if save_fig:
            print("[WARNING]: 'show_fig' is incompatible with 'save_fig'")
    else:
        # Close figure
        plt.close(fig) if fig else plt.close()


def do_all_figs_exists(output_dir, fname, formats):
    for ext in formats:
        # Create png/pdf/... dirs
        save_dir = os.path.join(output_dir, ext)
        if os.path.exists(os.path.join(save_dir, f"{fname}.{ext}")):
            return True
    return False


def plot_metrics(output_path, df_report, plot_metric, save_figures=True, show_figures=False):
    # Check if the metric id is in the dataframe
    if plot_metric not in df_report.columns:
        raise ValueError(f"Metric '{plot_metric}' was not found in the given dataframe")

    # Set backend
    if save_figures:
        set_non_gui_backend()
        if show_figures:
            raise ValueError("'save_fig' is incompatible with 'show_fig'")

    print(f"=> Plotting metrics...")
    print(f"   [WARNING]: Matplotlib might miss some images if the loop is too fast")

    # Parse metric name
    beam_width, metric_name = plot_metric.split('__')
    beam_width = beam_width.replace("beam", "beam=")
    metric_name = metric_name.replace('_', ' ').replace("score", "").strip().title()

    # Set other values
    ylabel = f"{metric_name} ({beam_width})"
    fname = f"report__{plot_metric}"

    # Make run name smaller with break lines
    df_report["alias"] = [f"{row['subword_model']} - {row['vocab_size']}\nTr: {row['train_dataset'].replace('_', ' ')}" for i, row in df_report.iterrows()]

    # Evals per model
    num_same_tr_ts = int((df_report['train_dataset'] == df_report['eval_dataset']).values.sum())
    total_tr = len(df_report['train_dataset'])
    hue = None if num_same_tr_ts == total_tr else "eval_dataset"

    # Plot data
    width = 16  #min(max(total_tr, 8), 24)
    height = 8
    catplot(data=df_report, x="alias", y=plot_metric, hue=hue,
            title=f"Model comparison", xlabel="Models", ylabel=ylabel, leyend_title=None,
            output_dir=output_path, fname=fname, aspect_ratio=(width, height), size=1.5, rotate_xlabels=90,
            save_fig=save_figures, show_fig=show_figures, overwrite=True, data_format="{:.2f}", loc="lower right")


def plot_vocabs_report(output_path, data, x, y_left, y_right=None, loc_legend="upper left", prefix="", save_figures=True, show_figures=False):
    # Check if the metrics are in the dataframe
    y_left, y_left_hue = y_left if isinstance(y_left, tuple) else (y_left, None)
    if y_left not in data.columns:
        raise ValueError(f"'{y_left}' was not found in the given dataframe")

    # Check if the metrics are in the dataframe
    y_right, y_right_hue = y_right if isinstance(y_right, tuple) else (y_right, None)
    if y_right and y_right not in data.columns:
        raise ValueError(f"'{y_right}' was not found in the given dataframe")

    # Set backend
    if save_figures:
        set_non_gui_backend()
        if show_figures:
            raise ValueError("'save_fig' is incompatible with 'show_fig'")

    print(f"=> Plotting vocabs report...")
    print(f"   [WARNING]: Matplotlib might miss some images if the loop is too fast")

    # Set other values
    title = f"{y_left.title()} {'and ' + y_right.title() if y_right else ''} depending on the {x.title()}".replace('_', ' ').replace('  ', ' ')
    xlabel = "Vocab sizes"
    ylabel_left = f"{y_left}".title().replace('_', ' ')
    ylabel_right = f"{y_right}".title().replace('_', ' ') if y_right else None
    fname = f"{prefix}vocabs_report__{y_left}{'_' + y_right if y_right else ''}".lower()

    # Plot data
    lineplot(data=data, x=x,
             y_left=y_left, y_left_hue=y_left_hue,
             y_right=y_right, y_right_hue=y_right_hue,
             title=title, xlabel=xlabel, ylabel_left=ylabel_left, ylabel_right=ylabel_right,
             leyend_title=None, output_dir=output_path, fname=fname, aspect_ratio=(8, 6), size=1.0, rotate_xlabels=0,
             save_fig=save_figures, show_fig=show_figures, overwrite=True, data_format="{:.2f}", loc_legend=loc_legend)
