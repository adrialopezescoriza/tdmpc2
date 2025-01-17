import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.font_manager as fm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from results import *

MAX_STEPS = 500
PLOT_STEP = 1  # * 1e3

TASKS_DEMOS = {
    "Maniskill-Manipulation": {
        'stack-cube': [25],
        'peg-insertion': [100],
        'lift-peg-upright': [5],
        'poke-cube': [5],
        'pick-place': [100],
    },
    "Metaworld": {
        'mw-assembly': [5],
        'mw-peg-insert-side': [5],
        'mw-pick-place': [5],
        'mw-stick-push': [5],
        'mw-stick-pull': [5],
    },
    "Maniskill-Humanoids": {
        'humanoid-place-apple': [5],
        'humanoid-transport-box': [50],
    },
    "Robosuite": {
        'robosuite-lift': [5],
        'robosuite-door': [10],
        'robosuite-pick-place-can': [20],
        'robosuite-stack': [10],
    }
}

ALGORITHMS = [
    "Modem2 + DrS",
    "TDMPC2",
    "Modem",
    "LaNE",
]

# Define a dictionary with custom x-axis limits for each domain
MAX_STEPS_DICT = {
    "Maniskill-Manipulation": 500,
    "Metaworld": 500,
    "Maniskill-Humanoids": 100,
    "Robosuite": 100,
}

PLOT_STEP_DICT = {
    "Maniskill-Manipulation": 5,
    "Metaworld": 5,
    "Maniskill-Humanoids": 1,
    "Robosuite": 1,
}

def pad_to_max_steps(df, max_steps, step_col='step', value_col='success'):
    """
    Pads the DataFrame to max_steps using the average of the last 50 values.
    """
    if df[step_col].max() < max_steps:
        # Compute the average of the last 50 values
        last_5_avg = df[value_col].iloc[-5:].mean() if len(df) >= 5 else df[value_col].mean()
        # Generate missing steps
        missing_steps = np.arange(df[step_col].max(), max_steps, PLOT_STEP)
        padding = pd.DataFrame({
            step_col: missing_steps,
            value_col: [last_5_avg] * len(missing_steps)
        })
        # Concatenate the original DataFrame with the padding
        df = pd.concat([df, padding], ignore_index=True)
    return df

def main():
    set_style()

    exp_names = ALGORITHMS
    domain_results = defaultdict(lambda: defaultdict(list))

    # Process results for each domain and task
    for domain, tasks_demos in TASKS_DEMOS.items():
        max_steps = MAX_STEPS_DICT.get(domain, MAX_STEPS)  # Get domain-specific max_steps
        tasks = list(tasks_demos.keys())
        exp_name_to_runs = {
            exp_name: {
                task: get_results(PATH / 'csv' / ALGO_TO_LABEL[exp_name] / f'{task}-semi.csv') for task in tasks
            } for exp_name in exp_names
        }

        for exp_name in exp_names:
            results = exp_name_to_runs[exp_name]
            for task, demos in tasks_demos.items():
                df = results[task]
                if df is None:
                    continue
                df = df[df['n_demos'].isin(demos + [0])].copy()
                df = pad_to_max_steps(df, max_steps)
                df = df[df['step'] % (MAX_STEPS_DICT[domain] / (100 * PLOT_STEP)) == 0]  # Filter for PLOT_STEP
                df['success'] = df['success'] * 100
                df['task'] = task
                domain_results[domain][exp_name].append(df)

    # Calculate domain averages
    domain_averages = defaultdict(dict)
    for domain, algo_results in domain_results.items():
        for exp_name, task_dfs in algo_results.items():
            combined_df = pd.concat(task_dfs, ignore_index=True)
            avg_df = combined_df.groupby(['step', 'seed', 'task']).agg({'success': 'mean'}).reset_index()
            domain_averages[domain][exp_name] = avg_df

    # Plot domain results
    f, axs = plt.subplots(1, 4, figsize=(36, 6), sharey=True)
    axs = axs.flatten()
    domains = list(TASKS_DEMOS.keys())

    for i, domain in enumerate(domains):
        ax = axs[i]
        max_steps = MAX_STEPS_DICT.get(domain, MAX_STEPS)  # Domain-specific max_steps
        algo_results = domain_averages[domain]

        for exp_name, avg_df in algo_results.items():
            sns.lineplot(
                x='step',
                y='success',
                data=avg_df,
                ax=ax,
                errorbar=('ci', 95),
                legend=False,
                label=ALGO_TO_LABEL.get(exp_name, exp_name),
                color=COLORS[ALGO_TO_COLOR[exp_name]],
                linewidth=4 if ALGO_TO_LABEL[exp_name] == 'Ours' else 3,
                err_kws={'alpha': 0.1},
            )
        ax.set_title(domain.replace('goto', 'reach').replace('-corridor', '').replace('corridor', 'run').replace('-', ' ').title(), fontsize=30)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xlim(0, max_steps)  # Use domain-specific max_steps
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}' + ('K' if x > 0 else '')))
        ax.xaxis.set_major_locator(plt.MultipleLocator(max_steps / 2))
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0f}%'))
        ax.yaxis.set_major_locator(plt.MultipleLocator(50))

        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
    
    #f.supxlabel("Interaction Steps", fontsize=30)

    h, l = [], []
    for ax in axs:
        _h, _l = ax.get_legend_handles_labels()
        if len(_h) > len(h):
            h, l = _h, _l
    
    # Update the font properties for "Ours"
    legend_labels = []
    font_properties = []
    for label in l:
        if label == "Ours":
            # Use a bold font for "Ours"
            font_properties.append(fm.FontProperties(weight="bold", size=30))
        else:
            # Use the default font for other labels
            font_properties.append(fm.FontProperties(size=30))
        legend_labels.append(label)

    # Add the custom legend to the figure
    legend = f.legend(h, legend_labels, loc="lower center", ncol=len(exp_names), frameon=False)
    for text, font in zip(legend.get_texts(), font_properties):
        text.set_font_properties(font)

    f.subplots_adjust(bottom=0.24, wspace=0.15, hspace=0.375)
    save_fig('overall')

if __name__ == '__main__':
    main()
