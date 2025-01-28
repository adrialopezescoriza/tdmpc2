import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import matplotlib.font_manager as fm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from results import *

MAX_STEPS = 500
PLOT_STEP = 1

TASKS_DEMOS_MANISKILL = {
    'stack-cube': [5, 10, 25, 50, 100, 200],
    'peg-insertion': [5, 25, 50, 100, 200],
}

ALGORITHMS = [
    "Modem2 + DrS",
    "Modem2",
    "TDMPC2 + DrS",
    "Modem",
]

TASKS_DEMOS = TASKS_DEMOS_MANISKILL

def compute_step_to_reach_30(df, n_demos, algorithm):
    """Compute the step at which success rate crosses 30% for each seed."""
    results = []
    for seed, seed_df in df.groupby('seed'):
        seed_avg = seed_df.groupby('step').agg({'success': 'mean'}).reset_index()
        above_30 = seed_avg[seed_avg['success'] >= 30]
        if not above_30.empty:
            step = above_30.iloc[0]['step']
        else:
            step = MAX_STEPS  # If 30% is never reached
        results.append({
            'algorithm': algorithm,
            'n_demos': n_demos,
            'seed': seed,
            'step_to_30': step,
        })
    return pd.DataFrame(results)

def main():
    set_style()

    tasks = list(TASKS_DEMOS.keys())
    exp_names = ALGORITHMS
    combined_results = []

    # Process results for each algorithm
    for exp_name in exp_names:
        for task, n_demos_list in TASKS_DEMOS.items():
            # Fetch results for each task
            df = get_results(PATH / 'csv' / ALGO_TO_LABEL[exp_name] / f'{task}-semi.csv')
            if df is None:
                continue
            df = df[df['n_demos'].isin(n_demos_list + [0])].copy()
            df['success'] = df['success'] * 100  # Convert success to percentage
            df['algorithm'] = ALGO_TO_LABEL.get(exp_name, exp_name)  # Add algorithm label

            # Compute steps to reach 30% for each n_demos
            for n_demos in n_demos_list:
                demo_specific_df = df[df['n_demos'] == n_demos]
                demo_specific_df = demo_specific_df[demo_specific_df['step'] % PLOT_STEP == 0]
                demo_results = compute_step_to_reach_30(demo_specific_df, n_demos, ALGO_TO_LABEL[exp_name])
                combined_results.append(demo_results)

    # Combine all results into a single DataFrame for seaborn
    combined_results = pd.concat(combined_results, ignore_index=True)

    # Convert n_demos to a categorical variable for equal spacing
    combined_results['n_demos'] = combined_results['n_demos'].astype(str)

    # Plot using seaborn
    plt.figure(figsize=(19, 14))

    # Make "Ours" line bold
    for exp_name in exp_names:
        sns.lineplot(
            data=combined_results[combined_results['algorithm'] == ALGO_TO_LABEL.get(exp_name, exp_name)],
            x='n_demos',
            y='step_to_30',
            legend=False,
            label=ALGO_TO_LABEL.get(exp_name, exp_name),
            color=COLORS[ALGO_TO_COLOR[exp_name]],
            linewidth=4 if ALGO_TO_LABEL[exp_name] == 'Ours' else 3,
            errorbar=('ci', 95),
            err_kws={'alpha': 0.1},
        )

    # Formatting
    plt.xlabel('Number of Demos', fontsize=35)
    plt.ylabel('Steps to Reach 30% Success \u2193', fontsize=35)
    plt.xticks(fontsize=30)
    plt.xlim(0,5)
    plt.yticks(fontsize=30)
    plt.grid(True, linestyle='-', alpha=0.6)

    # Custom legend
    legend = plt.legend(
        title=None,
        fontsize=40,
        loc='upper center',
        bbox_to_anchor=(0.55, -0.12),  # Position legend below the plot
        ncol=len(ALGORITHMS) / 2,
        frameon=False,
        handleheight=0.5,
    )
    # Adjust line alignment in legend
    for handle in legend.legend_handles:
        handle.set_linewidth(10)

    # Bold "Ours" in legend
    for text in legend.get_texts():
        if text.get_text() == "Ours":
            text.set_font_properties(fm.FontProperties(weight="bold", size=40))

    # Save the figure
    plt.tight_layout()
    save_fig('aggregated_demos')

if __name__ == '__main__':
    main()
