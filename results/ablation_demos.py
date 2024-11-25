import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from results import *

MAX_STEPS = 500
PLOT_STEP = 10 # x 1e3

TASKS_DEMOS_MANISKILL = {
    # 'stack-cube': [1,5,25,100,100],
    'peg-insertion': [5,25,50,100,200],
    # 'lift-peg-upright': [1,5,25,100,100],
    # 'poke-cube': [1,5,25,100,100],
    # 'pick-place': [1,5,25,100,100],
}

TASKS_DEMOS_METAWORLD = {
    'mw-assembly': [1,5,25,50,100],
    'mw-peg-insert-side': [1,5,25,50,100],
    'mw-pick-place': [1,5,25,50,100],
    'mw-stick-push': [1,5,25,50,100],
    'mw-stick-pull': [1,5,25,50,100],
}

ALGORITHMS = [
    "Modem2 + DrS",
    #"Modem2",
    "TDMPC2",
    #"TDMPC2 + DrS",
    "Modem",
]

TASKS_DEMOS = TASKS_DEMOS_MANISKILL
# TASKS_DEMOS.update(TASKS_DEMOS_MANISKILL)

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

    # Process results for each algorithm
    for exp_name in exp_names:
        results = exp_name_to_runs[exp_name]

        # Combine and average across tasks for each `n_demos`
        all_demos_results = {}
        for task in tasks:
            df = results[task]
            df = df[df['n_demos'].isin(TASKS_DEMOS[task] + [0])].copy() if df is not None else None
            if df is None:
                continue
            df['success'] = df['success'] * 100
            df['task'] = task

            # Group by number of demos and step, average across tasks
            for n_demos in TASKS_DEMOS[task]:
                if n_demos not in all_demos_results:
                    all_demos_results[n_demos] = []
                demo_specific_df = df[df['n_demos'] == n_demos]

                # Filter only steps that are multiples of 5000
                demo_specific_df = demo_specific_df[demo_specific_df['step'] % PLOT_STEP == 0]
                all_demos_results[n_demos].append(demo_specific_df)

        # Aggregate results for each number of demos
        results['demos'] = {}
        for n_demos, demo_dfs in all_demos_results.items():
            combined_demo_results = pd.concat(demo_dfs, ignore_index=True)
            results['demos'][n_demos] = combined_demo_results.groupby(['step', 'seed']).agg({'success': 'mean'}).reset_index()

    # Plot results
    f, axs = plt.subplots(2, 3, figsize=(18, 6), sharex=True, sharey=True)
    axs = axs.flatten()

    demo_values = sorted({demo for task_demos in TASKS_DEMOS.values() for demo in task_demos})
    demo_labels = [f'{demo} demos' for demo in demo_values]

    for idx, n_demos in enumerate(demo_values):
        ax = axs[idx]

        for exp_name in exp_names:
            results = exp_name_to_runs[exp_name]
            if n_demos not in results['demos']:
                continue

            df = results['demos'][n_demos]
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

        ax.set_title(f'{n_demos} demos')
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xlim(0, MAX_STEPS)
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}' + ('K' if x > 0 else '')))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0f}%'))
        ax.yaxis.set_major_locator(plt.MultipleLocator(50))
        ax.xaxis.set_major_locator(plt.MultipleLocator(MAX_STEPS / 2))

    h, l = [], []
    for ax in axs:
        _h, _l = ax.get_legend_handles_labels()
        if len(_h) > len(h):
            h, l = _h, _l
    f.legend(h, l, loc='lower center', ncol=len(exp_names), frameon=False)
    f.subplots_adjust(bottom=0.185, wspace=0.15, hspace=0.375)
    save_fig('demos_ablation')


if __name__ == '__main__':
    main()