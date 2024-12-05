import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from results import *

MAX_STEPS = 200
PLOT_STEP = 1

TASKS_DEMOS_MANISKILL = {
    #'stack-cube': [5, 25, 50, 100, 200],
    #'peg-insertion': [5, 25, 100, 200],
    'lift-peg-upright': [1,5,25],
    #'poke-cube': [5, 25, 100, 200],
    #'pick-place': [5, 25,50, 100],
}

TASKS_DEMOS_METAWORLD = {
    'mw-assembly': [1, 5, 10, 20, 50],
    'mw-peg-insert-side': [1, 5, 10, 20, 50],
    'mw-pick-place': [1, 5, 10, 20, 50],
    #'mw-stick-push': [1, 5, 10, 20, 50],
    #'mw-stick-pull': [1, 5, 10, 20, 50],
}

ALGORITHMS = [
    "Modem2 + DrS",
    #"Modem2",
    # "TDMPC2",
    #"TDMPC2 + DrS",
    #"Modem",
]

TASKS_DEMOS = TASKS_DEMOS_METAWORLD
#TASKS_DEMOS.update(TASKS_DEMOS_MANISKILL)

def compute_step_to_reach_50(df):
    """Compute the step at which success rate crosses 50%."""
    df_avg = df.groupby('step').agg({'success': 'mean'}).reset_index()
    above_50 = df_avg[df_avg['success'] >= 50]
    if not above_50.empty:
        return above_50.iloc[0]['step']  # First step where success crosses 50%
    return None  # If 50% is never reached

def main():
    set_style()

    tasks = list(TASKS_DEMOS.keys())
    print('Tasks:', tasks)
    print('Number of tasks:', len(tasks))

    exp_names = ALGORITHMS
    exp_name_to_runs = {exp_name: {
        task: get_results(PATH / 'csv' / ALGO_TO_LABEL[exp_name] / f'{task}-semi.csv') for task in tasks
    } for exp_name in exp_names}

    step_to_reach_50 = defaultdict(dict)  # Store steps to reach 50% for each exp_name and demo count

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

                # Filter only steps that are multiples of PLOT_STEP
                demo_specific_df = demo_specific_df[demo_specific_df['step'] % PLOT_STEP == 0]
                all_demos_results[n_demos].append(demo_specific_df)

        # Aggregate results for each number of demos
        for n_demos, demo_dfs in all_demos_results.items():
            combined_demo_results = pd.concat(demo_dfs, ignore_index=True)
            demo_avg = combined_demo_results.groupby(['step', 'seed']).agg({'success': 'mean'}).reset_index()
            step = compute_step_to_reach_50(demo_avg)
            step_to_reach_50[exp_name][n_demos] = step

    # Plot step to reach 50% success
    plt.figure(figsize=(10, 6))
    demo_values = sorted({demo for task_demos in TASKS_DEMOS.values() for demo in task_demos})

    # Create equidistant x-ticks
    x_ticks = range(len(demo_values))

    for exp_name in exp_names:
        steps = [step_to_reach_50[exp_name].get(demo, None) for demo in demo_values]
        plt.plot(
            x_ticks,
            steps,
            label=ALGO_TO_LABEL.get(exp_name, exp_name),
            marker='o',
            linewidth=3,
        )

    plt.title('Steps to Reach 50% Success')
    plt.xlabel('Number of Demos')
    plt.ylabel('Steps (1e3)')
    plt.ylim(0, MAX_STEPS)
    plt.xticks(x_ticks, labels=demo_values)  # Assign demo values to equidistant ticks
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    save_fig('aggregated_demos')

if __name__ == '__main__':
    main()

