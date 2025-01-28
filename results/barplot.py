import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.font_manager as fm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from results import *

# Task and algorithm setup from overall.py
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

def compute_domain_averages():
    domain_results = defaultdict(lambda: defaultdict(list))

    # Process results for each domain and task
    for domain, tasks_demos in TASKS_DEMOS.items():
        tasks = list(tasks_demos.keys())
        exp_name_to_runs = {
            exp_name: {
                task: get_results(PATH / 'csv' / ALGO_TO_LABEL[exp_name] / f'{task}-semi.csv') for task in tasks
            } for exp_name in ALGORITHMS
        }

        for exp_name in ALGORITHMS:
            results = exp_name_to_runs[exp_name]
            for task, demos in tasks_demos.items():
                df = results[task]
                if df is None:
                    continue
                df = df[df['n_demos'].isin(demos + [0])].copy()
                df = df.groupby(['step', 'seed']).agg({'success': 'mean'}).reset_index()
                last_steps = df.loc[df.groupby('seed')['step'].idxmax()]
                average_success_rate = last_steps['success'].mean() if not df.empty else 0
                domain_results[domain][exp_name].append(average_success_rate)

    # Compute averages across tasks for each domain and algorithm
    domain_averages = {
        domain: {
            algo: 100 * sum(success_rates) / len(success_rates) if success_rates else 0
            for algo, success_rates in algo_results.items()
        }
        for domain, algo_results in domain_results.items()
    }

    return domain_averages

def plot_barplot(domain_averages):
    domains = list(domain_averages.keys())
    fig, axs = plt.subplots(1, 4, figsize=(36, 8), sharey=True)

    for i, domain in enumerate(domains):
        ax = axs[i]
        domain_data = domain_averages[domain]
        df = pd.DataFrame(list(domain_data.items()), columns=['Algorithm', 'Average Success Rate'])
        df['Algorithm Label'] = df['Algorithm'].map(ALGO_TO_LABEL)

        sns.barplot(
            data=df, 
            x='Algorithm Label', 
            y='Average Success Rate', 
            ax=ax, 
            palette=[COLORS[ALGO_TO_COLOR[algo]] for algo in df['Algorithm']],
            width=0.9,  # Adjust bar width to reduce whitespace
            legend=False,
        )

        # Add success rate on top of each bar
        for bar, (success_rate, algo_label) in zip(ax.patches, zip(df['Average Success Rate'], df['Algorithm Label'])):
            fontweight = "bold" if algo_label == "Ours" else "normal"
            ax.text(
                bar.get_x() + bar.get_width() / 2, 
                bar.get_height() + 1, 
                f'{success_rate:.1f}', 
                ha='center', 
                va='bottom', 
                fontsize=30,
                fontweight=fontweight
            )

        ax.set_title(domain.replace('-', ' ').title(), fontsize=36, weight="bold")
        ax.set_xlabel(None)
        ax.set_xticks([])  # Remove x-axis labels
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel('Success Rate (%)' if i == 0 else None, fontsize=36)
        ax.tick_params(axis='y', labelsize=30)
        ax.set_ylim(0, 110)

    # Create the legend
    handles = [plt.Line2D([0], [0], color=COLORS[ALGO_TO_COLOR[algo]], lw=4) for algo in ALGORITHMS]
    labels = [ALGO_TO_LABEL[algo] for algo in ALGORITHMS]

    # Custom legend with better spacing and alignment
    legend = fig.legend(
        handles, 
        labels, 
        loc="lower center", 
        bbox_to_anchor=(0.5, 0.1),  # Center legend horizontally below the subplots
        ncol=len(ALGORITHMS),  # Span horizontally
        frameon=False, 
        handletextpad=2.5,  # Increase spacing between legend line and text
        columnspacing=6.0,  # Increase spacing between legend columns
        handlelength=6.0,
        handleheight=1.5,
    )

    # Adjust line alignment in legend
    for handle in legend.legend_handles:
        handle.set_linewidth(10)

    # Adjust font properties for legend labels
    for text, label in zip(legend.get_texts(), labels):
        text.set_font_properties(fm.FontProperties(size=36, weight="bold" if label == "Ours" else "normal"))

    # Adjust subplot spacing
    fig.subplots_adjust(bottom=0.3, wspace=0.02, hspace=0.2)
    save_fig('barplot')


def main():
    domain_averages = compute_domain_averages()
    plot_barplot(domain_averages)

if __name__ == '__main__':
    main()