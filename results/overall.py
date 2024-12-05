import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from results import *

MAX_STEPS = 500
PLOT_STEP = 4  # * 1e3

TASKS_DEMOS = {
    "Maniskill": {
        'stack-cube': [25],
        'peg-insertion': [100],
        'lift-peg-upright': [5],
        'poke-cube': [5],
        'pick-place': [100],
    },
    "Humanoids": {
        'humanoid-place-apple': [5],
        'humanoid-transport-box': [5],
    },
    "Metaworld": {
        'mw-assembly': [5],
        'mw-peg-insert-side': [5],
        'mw-pick-place': [5],
        'mw-stick-push': [5],
        'mw-stick-pull': [5],
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
]

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
                df = pad_to_max_steps(df, MAX_STEPS)
                df = df[df['step'] % PLOT_STEP == 0]  # Filter for PLOT_STEP
                df['success'] = df['success'] * 100
                df['task'] = task
                # Pad the data to max steps
                domain_results[domain][exp_name].append(df)

    # Calculate domain averages
    domain_averages = defaultdict(dict)
    for domain, algo_results in domain_results.items():
        for exp_name, task_dfs in algo_results.items():
            combined_df = pd.concat(task_dfs, ignore_index=True)
            avg_df = combined_df.groupby(['step', 'seed', 'task']).agg({'success': 'mean'}).reset_index()
            domain_averages[domain][exp_name] = avg_df

    # Calculate overall average
    overall_averages = defaultdict(list)
    for domain, algo_results in domain_averages.items():
        for exp_name, avg_df in algo_results.items():
            overall_averages[exp_name].append(avg_df)
    overall_averages = {
        exp_name: pd.concat(dfs, ignore_index=True).groupby(['step', 'seed', 'task']).agg({'success': 'mean'}).reset_index()
        for exp_name, dfs in overall_averages.items()
    }

    # Plot domain results
    f, axs = plt.subplots(2, 3, figsize=(18, 6), sharex=True, sharey=True)
    axs = axs.flatten()
    domains = ['Average'] + list(TASKS_DEMOS.keys())

    for i, domain in enumerate(domains):
        ax = axs[i]
        if domain == 'Average':
            algo_results = overall_averages
        else:
            algo_results = domain_averages[domain]
        
        for exp_name, avg_df in algo_results.items():
            sns.lineplot(
                x='step',
                y='success',
                data=avg_df,
                ax=ax,
                errorbar=('ci', 75),
                legend=False,
                label=ALGO_TO_LABEL.get(exp_name, exp_name),
                color=COLORS[ALGO_TO_COLOR[exp_name]],
                linewidth=4 if ALGO_TO_LABEL[exp_name] == 'Ours' else 3,
                err_kws={'alpha': 0.1},
            )
        # make title bold if average
        if task == 'Average':
            ax.set_title(domain.title(), fontweight='bold')
        else:
            ax.set_title(domain.replace('goto', 'reach').replace('-corridor', '').replace('corridor', 'run').replace('-', ' ').title())
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xlim(0, MAX_STEPS)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}' + ('K' if x > 0 else '')))
        ax.xaxis.set_major_locator(plt.MultipleLocator(MAX_STEPS / 2))
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0f}%'))
        ax.yaxis.set_major_locator(plt.MultipleLocator(50))

    h, l = [], []
    for ax in axs:
        _h, _l = ax.get_legend_handles_labels()
        if len(_h) > len(h):
            h, l = _h, _l
    f.legend(h, l, loc='lower center', ncol=len(exp_names), frameon=False)
    f.subplots_adjust(bottom=0.185, wspace=0.15, hspace=0.375)
    save_fig('domain_averages')

if __name__ == '__main__':
    main()



