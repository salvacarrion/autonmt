import os
from pathlib import Path

from autonmt import utils

# Sort of incompatible with plotly
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def set_non_gui_backend():
    # If this is not set, matplotlib won't most of the images that were created a loop
    # Source: https://github.com/matplotlib/matplotlib/issues/8519
    matplotlib.use('agg')


def catplot(data, x, y, hue, title, xlabel, ylabel, leyend_title, output_dir, fname, aspect_ratio=(12, 8), size=1.0,
            show_values=True, dpi=150, show_fig=False, save_fig=True, formats=None, overwrite=True, data_format='{:.0f}'):
    if formats is None:
        formats = ["png", "pdf"]

    # Check if the figures exists
    if not overwrite and (save_fig and do_all_figs_exists(output_dir, fname, formats)):
        print(f"\t\t\t- Skipped catplot as it already exists ({fname})")
        return False

    # Create subplot
    fig = plt.figure(figsize=(aspect_ratio[0] * size, aspect_ratio[1] * size))
    sns.set(font_scale=size)

    # Plot catplot
    g = sns.catplot(data=data, x=x, y=y, hue=hue, kind="bar", legend=False, height=aspect_ratio[1],
                    aspect=aspect_ratio[0] / aspect_ratio[1])

    # Tweaks
    g.set(xlabel=xlabel, ylabel=ylabel)
    # g.set_xticklabels(g.get_xticklabels(), rotation=90)
    # g.tick_params(axis='x', which='major', labelsize=8*size)
    # g.tick_params(axis='y', which='major', labelsize=8*size)
    g.axes.flat[0].yaxis.set_major_formatter(utils.human_format_int)  # only for catplot

    # Add values
    if show_values:
        ax = g.facet_axis(0, 0)
        for c in ax.containers:
            labels = [data_format.format(float(v.get_height())) for v in c]
            ax.bar_label(c, labels=labels, label_type='edge', fontsize=8 * size)

    # properties
    g.set(xlabel=xlabel, ylabel=ylabel)
    plt.title(title)
    plt.legend(title=leyend_title, loc='upper right')
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


def plot_metrics(output_path, df_metrics, metric_id, save_figures, show_figures):
    # Set backend
    if save_figures:
        set_non_gui_backend()
        if show_figures:
            raise ValueError("'save_fig' is incompatible with 'show_fig'")

    print(f"=> Plotting metrics...")
    print(f"   [WARNING]: Matplotlib might miss some images if the loop is too fast")

    metric_name = metric_id.split('_')[-1].upper()
    p_fname = f"metrics__{metric_name}"

    # Check if the metric id is in the dataframe
    if metric_id not in df_metrics.columns:
        raise ValueError(f"Metric '{metric_id}' was not found in the given dataframe")

    # Plot data
    catplot(data=df_metrics, x="run_name", y=metric_id, hue="eval_dataset",
                  title=f"Model comparison", xlabel="Models", ylabel=metric_name, leyend_title=None,
                  output_dir=output_path, fname=p_fname, aspect_ratio=(8, 4), size=1.0,
                  save_fig=save_figures, show_fig=show_figures, overwrite=True, data_format="{:.2f}")
    
# def histogram(data, output_dir, fname, title="", legend_title=None, labels=None, nbins=20, bargap=0.2,
#                show_fig=False, save_fig=True, formats=None):
#     import plotly.express as px
#     import plotly.graph_objects as go
#
#     # Fix mathjax issue
#     import plotly.io as pio
#     pio.kaleido.scope.mathjax = None
#
#     if formats is None:
#         formats = ["png", "pdf"]
#
#     fig = px.histogram(data, nbins=nbins, title=title, labels=labels)
#     fig.update_layout(bargap=bargap, legend_orientation='h', title=dict(x=0.5, font=dict(size=18)),
#                       legend_title=legend_title)
#
#     # Save image
#     if save_fig:
#         for ext in formats:
#             # Create png/pdf/... dirs
#             save_dir = os.path.join(output_dir, ext)
#             Path(save_dir).mkdir(parents=True, exist_ok=True)
#
#             # Save image
#             fig.write_image(os.path.join(save_dir, f"{fname}.{ext}"))
