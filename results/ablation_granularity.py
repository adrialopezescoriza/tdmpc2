import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from results import *

MAX_STEPS = 200
PLOT_STEP = 2

TASKS = {
    'stack-cube': [25],
    'stack-cube-semi': [25],
    'stack-cube-2-stages-semi': [25],
    'stack-cube-1-stages-semi': [25],
    # 'peg-insertion': [100],
    # 'peg-insertion-semi': [100],
    # 'peg-insertion-2-stages-semi': [100],
    # 'peg-insertion-1-stages-semi': [100],
}

N_STAGES = {
    'stack-cube-semi': "3 stages",
    'stack-cube-1-stages-semi': "1 stage",
    'stack-cube-2-stages-semi': "2 stages",
    'stack-cube': "Dense",
    'peg-insertion-semi': "3 stages",
    'peg-insertion-1-stages-semi': "1 stage",
    'peg-insertion-2-stages-semi': "2 stages",
    'peg-insertion': "Dense",
}

STAGES_TO_COLORS = {
    "Dense": 3,
    "1 stage": 4,
    "2 stages": 5,
    "3 stages": 0,
}

TASK_TO_ALGO = {
    'stack-cube-semi': 'Ours',
    'stack-cube-1-stages-semi': 'Ours',
    'stack-cube-2-stages-semi': 'Ours',
    'stack-cube': 'no learned reward',
    'peg-insertion-semi': 'Ours',
    'peg-insertion-1-stages-semi': 'Ours',
    'peg-insertion-2-stages-semi': 'Ours',
    'peg-insertion': 'no learned reward',
}

def main():
    set_style()

    tasks = list(TASKS.keys())
    stage_results = defaultdict(list)

    for task, demos in TASKS.items():
        n_stages = N_STAGES[task]
        df = get_results(PATH / 'csv' / TASK_TO_ALGO[task] / f'{task}.csv')
        if df is not None:
            df = df[df['n_demos'].isin(demos)]
            df = df[df['step'] % PLOT_STEP == 0]
            df['success'] = df['success'] * 100
            df['stage'] = n_stages
            stage_results[n_stages].append(df)

    aggregated_results = {}
    for stage, dfs in stage_results.items():
        combined_df = pd.concat(dfs, ignore_index=True)

        # Aggregate success across tasks for each seed
        task_avg_df = combined_df.groupby(['step', 'seed']).agg({
            'success': 'mean'
        }).reset_index()

        # Add to the results for plotting
        task_avg_df['stage'] = stage
        aggregated_results[stage] = task_avg_df

    # Combine all stages for a single seaborn lineplot call
    all_data = pd.concat(aggregated_results.values(), ignore_index=True)

    plt.figure(figsize=(19, 10))

    # Seaborn lineplot with automatic error bars
    sns.lineplot(
        x='step',
        y='success',
        hue='stage',
        data=all_data,
        palette={stage: COLORS[STAGES_TO_COLORS[stage]] for stage in STAGES_TO_COLORS.keys()},
        linewidth=5,
        legend=True,
        errorbar='ci',
        err_kws={'alpha':0.15},
    )

    # Formatting
    plt.xlabel('Interaction Steps', fontsize=35)
    plt.ylabel(None)
    plt.xticks(
        ticks=[0, MAX_STEPS // 2, MAX_STEPS],  # Set ticks at 0, middle, and max
        labels=['0', f'{MAX_STEPS // 2}K', f'{MAX_STEPS}K'],
        fontsize=35
    )
    plt.yticks(fontsize=35)
    plt.xlim(0, MAX_STEPS)
    plt.ylim(-2, 105)
    plt.grid(True, linestyle='-', alpha=0.6)
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: f'{y:.0f}%'))

    # Custom legend
    legend = plt.legend(
        title=None,
        fontsize=40,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.13),  # Position legend below the plot
        ncol=len(STAGES_TO_COLORS),
        frameon=False,
        handleheight=0.5,
    )
    # Adjust line alignment in legend
    for handle in legend.legend_handles:
        handle.set_linewidth(10)

    # Bold "Ours" in legend
    for text in legend.get_texts():
        if text.get_text() == "3 stages":
            text.set_font_properties(fm.FontProperties(weight="bold", size=40))

    # Save the figure
    plt.tight_layout()
    save_fig('ablation_granularity')

if __name__ == '__main__':
    main()