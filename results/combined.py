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
}

TASKS_DEMOS_HUMANOIDS = {
    'humanoid-place-apple': [5],
    'humanoid-transport-box': [50],
}

TASKS_DEMOS_METAWORLD = {
    'mw-stick-push': [5],
    'mw-stick-pull': [5],
}

TASKS_DEMOS_ROBOSUITE = {
    'robosuite-door': [10],
    'robosuite-pick-place-can': [20],
}

ALGORITHMS = [
    "Modem2 + DrS",
    "Modem",
    "TDMPC2",
    "LaNE",
]

SUITE_DICT = {
    "maniskill": TASKS_DEMOS_MANISKILL,
    "humanoids": TASKS_DEMOS_HUMANOIDS,
    "metaworld": TASKS_DEMOS_METAWORLD,
    "robosuite": TASKS_DEMOS_ROBOSUITE,
} 

MAX_STEPS_LEFT = 500
MAX_STEPS_RIGHT = 100
PLOT_STEP_LEFT = 5 # * 1e3
PLOT_STEP_RIGHT= 1 # * 1e3

def main():
    set_style()

    tasks = list(SUITE_DICT["maniskill"].keys()) + list(SUITE_DICT["metaworld"].keys()) + \
            list(SUITE_DICT["humanoids"].keys()) + list(SUITE_DICT["robosuite"].keys())

    exp_names = ALGORITHMS
    exp_name_to_runs = {exp_name: {
            task: get_results(PATH / 'csv' / ALGO_TO_LABEL[exp_name] / f'{task}-semi.csv') for task in tasks}
        for exp_name in exp_names
    }

    f, axs = plt.subplots(2, 4, figsize=(35, 12), sharex='col', sharey=True)
    axs = axs.flatten()

    for col, (suite, max_steps, plot_step) in enumerate(zip(["maniskill", "metaworld", "humanoids", "robosuite"],
                                               [MAX_STEPS_LEFT, MAX_STEPS_LEFT, MAX_STEPS_RIGHT, MAX_STEPS_RIGHT],
                                               [PLOT_STEP_LEFT, PLOT_STEP_LEFT, PLOT_STEP_RIGHT, PLOT_STEP_RIGHT])):
        tasks = list(SUITE_DICT[suite].keys())
        for exp_name in exp_names:
            results = exp_name_to_runs[exp_name]
            for row, task in enumerate(tasks):
                df = results.get(task)
                if df is None:
                    continue
                ax = axs[col + row * 4]
                # Filter rows based on step and n_demos
                df = df[df['n_demos'].isin(SUITE_DICT[suite][task] + [0])].copy()
                df = df[df['step'] % plot_step == 0]
                df['success'] = df['success'] * 100
                sns.lineplot(
                    x='step',
                    y='success',
                    data=df,
                    ax=ax,
                    errorbar=('ci', 95),
                    legend=False,
                    label=ALGO_TO_LABEL.get(exp_name, exp_name),
                    color=COLORS[ALGO_TO_COLOR[exp_name]],
                    linewidth=5 if ALGO_TO_LABEL[exp_name] == 'Ours' else 4,
                    err_kws={'alpha': 0.2},
                )
                ax.set_title(task.replace('mw-', '').replace('humanoid-', '').replace('robosuite-', '').replace('-', ' ').title(),
                             fontsize=30)

                ax.set_xlabel(None)
                ax.set_ylabel(None)
                ax.set_xlim(0, max_steps)
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}' + ('K' if x > 0 else '')))
                ax.xaxis.set_major_locator(plt.MultipleLocator(max_steps / 2))
                ax.set_ylim(-2, 100)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0f}%'))
                ax.yaxis.set_major_locator(plt.MultipleLocator(50))
                ax.tick_params(axis='x', labelsize=25)
                ax.tick_params(axis='y', labelsize=25)
    
    f.supxlabel("Interaction Steps", fontsize=25, y=0.07)
    
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
    legend = f.legend(
        h, 
        legend_labels, 
        loc="lower center", 
        bbox_to_anchor=(0.5, 0.0),  # Center legend horizontally below the subplots
        ncol=len(ALGORITHMS),  # Span horizontally
        frameon=False, 
        handleheight=1.5,
    )

    # Adjust line alignment in legend
    for handle in legend.legend_handles:
        handle.set_linewidth(6)

    for text, font in zip(legend.get_texts(), font_properties):
        text.set_font_properties(font)

    f.subplots_adjust(bottom=0.16, wspace=0.15, hspace=0.2)
    save_fig("combined")


if __name__ == '__main__':
    main()
