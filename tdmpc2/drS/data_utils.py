import numpy as np
import pickle
import torch

from tensordict.tensordict import TensorDict

def load_demo_dataset(path, keys=['observations', 'actions'], num_traj=None, success_only=False):
    with open(path, 'rb') as f:
        trajectories = pickle.load(f)
    if success_only:
        trajectories = [t for t in trajectories if t['infos'][-1]['success']]
    if num_traj is not None:
        trajectories = trajectories[:num_traj]
    # trajectories is a list of trajectory
    # trajectories[0] has keys like: ['actions', 'dones', ...]
    dataset = {}
    for key in keys:
        if key in ['observations', 'states'] and \
                len(trajectories[0][key]) > len(trajectories[0]['actions']):
            dataset[key] = np.concatenate([
                t[key][:-1] for t in trajectories
            ], axis=0)
        elif key[:5] == 'next_' and key not in trajectories[0]:
            if isinstance(trajectories[0][key[5:]], dict):
                dataset[key] = {
                    k: np.concatenate([
                        t[key[5:]][k][1:] for t in trajectories
                    ], axis=0) for k in trajectories[0][key[5:]].keys()
                }
            else:
                dataset[key] = np.concatenate([
                    t[key[5:]][1:] for t in trajectories
                ], axis=0)
        else:
            dataset[key] = np.concatenate([
                t[key] for t in trajectories
            ], axis=0)
    return dataset

# TODO: Make more general for all kinds of datasets
def load_dataset_as_td(path, num_traj=None, success_only=False):
    """
    Dataset must be stored as list of np.arrays
    Returns list of TensorDicts with different lengths

    trajectories is a list of trajectory
    trajectories[0] has keys like: ['actions', 'dones', ...]

    Note: next_observations means the observation resulting from a given action
    """
    with open(path, 'rb') as f:
        trajectories = pickle.load(f)
    if success_only:
        trajectories = [t for t in trajectories if t['infos'][-1]['success']]
    if num_traj is not None:
        assert num_traj <= len(trajectories), f"Trying to get {num_traj} from demo file with {len(trajectories)} demos"
        trajectories = trajectories[:num_traj]

    def episode_to_tensor(episode):
        if isinstance(episode[0], (torch.Tensor, TensorDict)):
            return torch.stack([o for o in episode])
        elif isinstance(episode[0], dict):
            return torch.stack([TensorDict(o) for o in episode])
        return torch.stack([torch.tensor(o) for o in episode])

    tds = []
    for traj in trajectories:
        tds.append(TensorDict(dict(
            obs=episode_to_tensor(traj['next_observations' if 'next_observations' in traj.keys() else 'observations']),
            reward=torch.tensor(traj['rewards']),
            action=episode_to_tensor(traj['actions']),
        ), batch_size=(len(traj['rewards']),)))

    return tds

def load_raw_trajectories(path, num_traj=None, success_only=False):
    with open(path, 'rb') as f:
        trajectories = pickle.load(f)
    if success_only:
        trajectories = [t for t in trajectories if t['infos'][-1]['success']]
    if num_traj is not None:
        trajectories = trajectories[:num_traj]
    # trajectories is a list of trajectory
    # trajectories[0] has keys like: ['actions', 'dones', ...]
    return trajectories

def sample_from_multi_buffers(buffers, batch_size):
    # Warning: when the buffers are full, this will make samples not uniform
    sizes = [b.size for b in buffers]
    tot_size = sum(sizes)
    if tot_size == 0:
        raise Exception('All buffers are empty!')
    n_samples = [int(s / tot_size * batch_size) for s in sizes]
    if sum(n_samples) < batch_size:
        n_samples[np.argmax(sizes)] += batch_size - sum(n_samples)
    batches = []
    for b, n in zip(buffers, n_samples):
        if n > 0:
            if b.size == 0:
                raise Exception('Buffer is empty!')
            batches.append(b.sample(n))
    ret = {}
    for k in batches[0].keys():
        ret[k] = torch.cat([b[k] for b in batches], dim=0)
    return ret