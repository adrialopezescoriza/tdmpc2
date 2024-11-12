import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from plots import *


SEEDS = [1,2,3]
# SEEDS = [1]


def results_to_csv(exp_name=None, group='eval', key='episode_success', steps=1_000_192):
    assert exp_name is not None, 'Please provide a valid `exp_name` argument.'
    exp_name = str(exp_name)

    runs = get_runs(entity=ENTITY, project=PROJECT)
    runs = filter_runs(runs, exp_name=exp_name)

    def get_avg_df(runs, group, key):
        if len(runs) == 0:
            return None
        new_key = key.replace('episode_', '')
        data = dict()
        for run in runs:
            df = run.history(keys=[f'{group}/{key}'], x_axis=f'{group}/step', pandas=True)
            df = df.rename(columns={f'{group}/{key}': new_key})
            objects = run.config.get('objects')
            seed = run.config.get('seed')
            df['seed'] = seed
            if len(df) == 0:
                continue
            if not objects in data:
                data[objects] = dict()
            if seed in SEEDS and (seed not in data[objects] or len(df) > len(data[objects][seed])):
                data[objects][seed] = df
    
        for objects in data.keys():
            data[objects] = pd.concat(data[objects].values(), ignore_index=True)

            # limit to plot range
            data[objects] = data[objects][data[objects][f'{group}/step'] <= steps]

            # round step to nearest 1e4
            # data[objects][f'{group}/step'] = data[objects][f'{group}/step'] / 1e4
            # data[objects][f'{group}/step'] = data[objects][f'{group}/step'].round(0) * 1e4

            # account for action repeat
            # data[objects]['eval/step'] = data[objects]['eval/step'] * 2

            # clean up
            data[objects] = data[objects].rename(columns={f'{group}/step': 'step'})
            data[objects]['step'] = data[objects]['step'].astype(int)
            if group == 'train': # average over steps
                data[objects] = data[objects].groupby(['step', 'seed']).mean().reset_index()
            data[objects][new_key] = data[objects][new_key].round(4)

            # warn if missing seeds or steps
            for seed in SEEDS:
                if seed not in data[objects]['seed'].values:
                    print(f'WARNING: missing seed {seed} for {objects}')
                elif len(data[objects][data[objects]['seed'] == seed]) < len(data[objects]) / len(SEEDS) - 1:
                    print(f'WARNING: missing steps for seed {seed} for {objects}')

        # filter out objects not in reference data
        # ref_objects = pd.read_csv('/data/nihansen/code/tdmpc2-turbo/plots/csv/mt100-ft.csv')
        # ref_objects = ref_objects['object'].unique()
        # data = {k: v for k, v in data.items() if int(k) in ref_objects}

        # save to csv
        fp = SAVE_PATH_CSV / f'{exp_name}.csv'
        fp.parent.mkdir(parents=True, exist_ok=True)
        # convert data to df and save to csv
        # add objects column
        df = pd.DataFrame()
        for objects in data.keys():
            data[objects]['objects'] = objects
            # make objects the first column
            df = pd.concat([df, data[objects]], ignore_index=True)
        # reorder columns
        df = df[['objects', 'seed', 'step', 'success']]
        # only keep last step for each seed
        df = df.groupby(['objects', 'seed']).last().reset_index()
        # warn if missing steps
        _df = df[df['step'] != steps]
        if len(_df) > 0:
            print('WARNING: missing steps for objects:\n', _df)
        # remove step column
        df = df.drop(columns=['step'])
        # rename objects to object
        df = df.rename(columns={'objects': 'object'})
        df.to_csv(fp, index=False)

        # print avg success rate
        print('Average success rate:', float(df['success'].mean()))

    df = get_avg_df(runs, group, key)


if __name__ == '__main__':
    results_to_csv(exp_name='auto-spec')
    # results_to_csv(exp_name='mt100-ft')
