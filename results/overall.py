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
    "ManiSkill Manipulation": {
        'stack-cube': [25],
        'peg-insertion': [100],
        'lift-peg-upright': [5],
        'poke-cube': [5],
        'pick-place': [100],
    },
    "Meta-World": {
        'mw-assembly': [5],
        'mw-peg-insert-side': [5],
        'mw-pick-place': [5],
        'mw-stick-push': [5],
        'mw-stick-pull': [5],
    },
    "ManiSkill Humanoids": {
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

MAX_STEPS_DICT = {
    "ManiSkill Manipulation": 500,
    "Meta-World": 500,
    "ManiSkill Humanoids": 100,
    "Robosuite": 100,
}

def pad_to_max_steps(df, max_steps, step_col='step', value_col='success'):
    if df[step_col].max() < max_steps:
        last_5_avg = df[value_col].iloc[-5:].mean() if len(df) >= 5 else df[value_col].mean()
        missing_steps = np.arange(df[step_col].max(), max_steps, PLOT_STEP)
        padding = pd.DataFrame({
            step_col: missing_steps,
            value_col: [last_5_avg] * len(missing_steps)
        })
        df = pd.concat([df, padding], ignore_index=True)
    return df

def load_bc_results(tasks_demos):
    """
    Load BC success values from the Pretraining folder.
    Returns: dict mapping task -> list of (seed, final success)
    """
    bc_results = defaultdict(list)
    for task, demos in tasks_demos.items():
        csv_path = PATH / 'csv' / 'Pretraining' / f'{task}-semi.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df = df[df['n_demos'].isin(demos + [0])].copy()
        df['success'] *= 100
        # Keep last success value per seed
        last_df = df.sort_values('step').groupby('seed').tail(1)
        for _, row in last_df.iterrows():
            bc_results[task].append((row['seed'], row['success']))
    return bc_results

def main():
    set_style()

    exp_names = ALGORITHMS
    domain_results = defaultdict(lambda: defaultdict(list))

    # Load BC data separately
    domain_bc_data = {}

    for domain, tasks_demos in TASKS_DEMOS.items():
        max_steps = MAX_STEPS_DICT.get(domain, MAX_STEPS)
        tasks = list(tasks_demos.keys())
        exp_name_to_runs = {
            exp_name: {
                task: get_results(PATH / 'csv' / ALGO_TO_LABEL[exp_name] / f'{task}-semi.csv') for task in tasks
            } for exp_name in exp_names
        }

        bc_task_results = load_bc_results(tasks_demos)
        domain_bc_data[domain] = bc_task_results

        for exp_name in exp_names:
            results = exp_name_to_runs[exp_name]
            for task, demos in tasks_demos.items():
                df = results[task]
                if df is None:
                    continue
                df = df[df['n_demos'].isin(demos + [0])].copy()
                df = pad_to_max_steps(df, max_steps)
                df = df[df['step'] % (MAX_STEPS_DICT[domain] / (100 * PLOT_STEP)) == 0]
                df['success'] *= 100
                df['task'] = task
                domain_results[domain][exp_name].append(df)

    domain_averages = defaultdict(dict)
    for domain, algo_results in domain_results.items():
        for exp_name, task_dfs in algo_results.items():
            combined_df = pd.concat(task_dfs, ignore_index=True)
            avg_df = combined_df.groupby(['step', 'seed', 'task']).agg({'success': 'mean'}).reset_index()
            domain_averages[domain][exp_name] = avg_df

    f, axs = plt.subplots(1, 4, figsize=(36, 8), sharey=True)
    axs = axs.flatten()
    domains = list(TASKS_DEMOS.keys())

    for i, domain in enumerate(domains):
        ax = axs[i]
        max_steps = MAX_STEPS_DICT.get(domain, MAX_STEPS)
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
                linewidth=5 if ALGO_TO_LABEL[exp_name] == 'Ours' else 4,
                err_kws={'alpha': 0.2},
            )

        # --- Plot Behavioral Cloning ---
        bc_entries = []
        steps = np.arange(0, max_steps + PLOT_STEP, PLOT_STEP)
        for task, records in domain_bc_data[domain].items():
            for seed, value in records:
                bc_entries.extend([{
                    'step': step,
                    'success': value,
                    'seed': seed,
                    'task': task
                } for step in steps])
        bc_df = pd.DataFrame(bc_entries)

        if not bc_df.empty:
            sns.lineplot(
                x="step",
                y="success",
                data=bc_df,
                ax=ax,
                errorbar=("ci", 95),
                legend=False,
                label="BC",
                color="k",
                linestyle="--",
                linewidth=3,
                err_kws={"alpha": 0.2}
            )

        ax.set_title(domain.replace('goto', 'reach').replace('-corridor', '').replace('corridor', 'run'), fontsize=33, weight="bold")
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xlim(0, max_steps)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}' + ('K' if x > 0 else '')))
        ax.xaxis.set_major_locator(plt.MultipleLocator(max_steps / 2))
        ax.set_ylim(-2, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0f}%'))
        ax.yaxis.set_major_locator(plt.MultipleLocator(50))
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

    f.supxlabel("Interaction Steps", fontsize=30, y=0.15)

    h, l = [], []
    for ax in axs:
        _h, _l = ax.get_legend_handles_labels()
        if len(_h) > len(h):
            h, l = _h, _l

    from matplotlib.lines import Line2D

    legend_labels = []
    font_properties = []
    for label in l:
        if label == "Ours":
            font_properties.append(fm.FontProperties(weight="bold", size=33))
        else:
            font_properties.append(fm.FontProperties(size=33))
        legend_labels.append(label)

    legend = f.legend(
        h,
        legend_labels,
        loc="lower center",
        ncol=len(exp_names) + 1,
        frameon=False,
        handleheight=2.0,
        handlelength=3.0  # <-- Increase this for longer lines
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(6)
    for text, font in zip(legend.get_texts(), font_properties):
        text.set_font_properties(font)

    f.subplots_adjust(bottom=0.3, wspace=0.15, hspace=0.375)
    save_fig('overall_bc')

if __name__ == '__main__':
    main()
