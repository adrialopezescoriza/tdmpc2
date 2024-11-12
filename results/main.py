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
from plots import *


EXP_TO_LABEL = {
    # 'sac': 'SAC',
    # 'sac-lowlevel-tdmpc2': 'SAC w/ LL TD-MPC$\\bf{2}$',
    'baseline': 'TD-MPC$\\bf{2}$',
    'default': 'Ours',
    # 'blind': 'Blind',
    # 'ft-corridor': 'Finetuned',
}
EXP_TO_COLOR = {
    # 'sac': -1,
    # 'sac-lowlevel-tdmpc2': 2,
    'baseline': 1,
    'default': 0,
    # 'blind': 3,
    # 'ft-corridor': 4,
}
MAX_RETURN = {
    'stand': 500,
    'walk': 500,
    'run': 500,
    'reach': 400,
    'corridor': 200,
    'hurdles-corridor': 200,
    'walls-corridor': 100,
    'gaps-corridor': 200,
    'stairs-corridor': 200,
}


def main():
    set_style()

    # tasks = TASKS
    tasks = [
        'stand',
        # 'walk',
        # 'run',
        # 'reach',
        'corridor',
        'hurdles-corridor',
        'walls-corridor',
        'gaps-corridor',
        'stairs-corridor',
        # 'pick-box',
    ]
    print('Tasks:', tasks)
    print('Number of tasks:', len(tasks))

    exp_names = EXP_TO_LABEL.keys()
    exp_name_to_runs = {exp_name: {
            task: get_results(PATH / 'csv' / exp_name / f'{task}.csv') for task in tasks}
        for exp_name in exp_names
    }

    # average over tasks
    for exp_name in exp_names:
        results = exp_name_to_runs[exp_name]
        for task in tasks:
            df = results[task]
            if df is None:
                continue
            df['reward'] = df['reward'] * 100 / MAX_RETURN[task]
            df['task'] = task
        results['average'] = pd.concat([df for df in results.values() if df is not None], ignore_index=True)
        results['average'] = results['average'].groupby(['step', 'seed']).agg({'reward': 'mean'}).reset_index()
    # prepend average to tasks
    tasks = ['average'] + tasks

    # f, axs = plt.subplots(1, 5, figsize=(18, 3.4), sharex=True, sharey=True)
    f, axs = plt.subplots(2, 5, figsize=(18, 6), sharex=True, sharey=True)
    axs = axs.flatten()

    # manually delete the last few subplots
    # f.delaxes(axs[-4])
    # f.delaxes(axs[-3])
    # f.delaxes(axs[-2])
    # f.delaxes(axs[-1])

    # Plot results
    for _, exp_name in enumerate(exp_names):
        results = exp_name_to_runs[exp_name]
        
        for j, task in enumerate(tasks):
            df = results[task]
            # if df is None:
            #     continue
            # df['reward'] = df['reward'] * 100 / MAX_RETURN[task]
            ax = axs[j]
            sns.lineplot(
                x='step',
                y='reward',
                data=df,
                ax=ax,
                errorbar=('ci', 95),
                legend=False,
                label=EXP_TO_LABEL.get(exp_name, exp_name),
                color=COLORS[EXP_TO_COLOR[exp_name]],
                linewidth=4 if exp_name == 'tdmpc2' else 3,
                err_kws={'alpha': 0.1},
            )
            # make title bold if average
            if task == 'average':
                ax.set_title(task.title(), fontweight='bold')
            else:
                ax.set_title(task.replace('goto', 'reach').replace('-corridor', '').replace('corridor', 'run').replace('-', ' ').title())
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_xlim(0, 3)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}' + ('M' if x > 0 else '')))
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
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
    save_fig('main')


if __name__ == '__main__':
    main()
