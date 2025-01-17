import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from results import *

TASKS_DEMOS_MANISKILL = {
    'stack-cube': [25],
    'peg-insertion': [100],
    'lift-peg-upright': [5],
    'poke-cube': [5],
    'pick-place': [100],
}

TASKS_DEMOS_HUMANOIDS = {
    'humanoid-place-apple': [5],
    'humanoid-transport-box': [50],
}

TASKS_DEMOS_METAWORLD = {
    'mw-assembly': [5],
    'mw-peg-insert-side': [5],
    'mw-pick-place': [5],
    'mw-stick-push': [5],
    'mw-stick-pull': [5],
}

TASKS_DEMOS_ROBOSUITE = {
    'robosuite-lift': [5],
    'robosuite-door': [10],
    'robosuite-pick-place-can': [20],
    'robosuite-stack': [10],
}

ALGORITHMS = [
    "Modem2 + DrS",
    "Modem",
    "TDMPC2",
    "Modem2",
    "TDMPC2 + DrS",
    #"LaNE",
]

SUITE_DICT = {
    "maniskill": TASKS_DEMOS_MANISKILL,
    "humanoids": TASKS_DEMOS_HUMANOIDS,
    "metaworld": TASKS_DEMOS_METAWORLD,
    "robosuite": TASKS_DEMOS_ROBOSUITE,
} 

SUITE = "maniskill"
TASKS_DEMOS = SUITE_DICT[SUITE] # TASKS_DEMOS_MANISKILL, TASKS_DEMOS_METAWORLD
# TASKS_DEMOS.update(TASKS_DEMOS_METAWORLD)
MAX_STEPS = 500
PLOT_STEP = 5 # * 1e3

def main():
    set_style()

    tasks = list(TASKS_DEMOS.keys())

    print('Tasks:', tasks)
    print('Number of tasks:', len(tasks))

    exp_names = ALGORITHMS
    exp_name_to_runs = {exp_name: {
            task: get_results(PATH / 'csv' / ALGO_TO_LABEL[exp_name] / f'{task}-semi.csv') for task in tasks}
        for exp_name in exp_names
    }

    for exp_name in exp_names:
        results = exp_name_to_runs[exp_name]
        for task in tasks:
            df = results[task]
            if df is None:
                continue
            # Filter rows based on step and n_demos
            df = df[df['n_demos'].isin(TASKS_DEMOS[task] + [0])].copy()
            df = df[df['step'] % PLOT_STEP == 0]  # Filter for PLOT_STEP
            df['success'] = df['success'] * 100
            df['task'] = task
            results[task] = df

        # Combine filtered data for averaging
        try:
            filtered_results = [df for df in results.values() if df is not None]
            results['average'] = pd.concat(filtered_results, ignore_index=True)
            results['average'] = results['average'].groupby(['step', 'seed']).agg({'success': 'mean'}).reset_index()
        except Exception as e:
            print(f"Error processing {exp_name}: {e}")

    # prepend average to tasks
    tasks = ['average'] + tasks

    #f, axs = plt.subplots(1, 3, figsize=(18, 3.4), sharex=False, sharey=True)
    #f, axs = plt.subplots(1, 3, figsize=(18, 3.4), sharex=True, sharey=True)
    f, axs = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    #f, axs = plt.subplots(2, 2, figsize=(18, 6), sharex=False, sharey=True)
    axs = axs.flatten()

    # Plot results
    for _, exp_name in enumerate(exp_names):
        results = exp_name_to_runs[exp_name]
        
        for j, task in enumerate(tasks):
            df = results[task]
            if df is None:
                continue
            ax = axs[j]
            sns.lineplot(
                x='step',
                y='success',
                data=df,
                ax=ax,
                errorbar=('ci', 95),
                legend=False,
                label=ALGO_TO_LABEL.get(exp_name, exp_name),
                color=COLORS[ALGO_TO_COLOR[exp_name]],
                linewidth=4 if ALGO_TO_LABEL[exp_name] == 'Ours' else 3,
                err_kws={'alpha': 0.1},
            )
            # make title bold if average
            if task == 'average':
                ax.set_title(task.title(), fontweight='bold', fontsize=30)
            else:
                ax.set_title(task.replace('goto', 'reach').replace('-corridor', '').replace('corridor', 'run').replace('mw-', '').replace('humanoid-','').replace('robosuite-','').replace('-', ' ').title(), fontsize=30)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_xlim(0, MAX_STEPS)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}' + ('K' if x > 0 else '')))
            ax.xaxis.set_major_locator(plt.MultipleLocator(MAX_STEPS / 2))
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0f}%'))
            ax.yaxis.set_major_locator(plt.MultipleLocator(50))

            ax.tick_params(axis='x', labelsize=25)
            ax.tick_params(axis='y', labelsize=25)
    
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

    f.subplots_adjust(bottom=0.15, wspace=0.15, hspace=0.25)
    save_fig('ablation_algorithms')


if __name__ == '__main__':
    main()
