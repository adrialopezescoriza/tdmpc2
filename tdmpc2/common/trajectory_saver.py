import os

class BaseTrajectorySaver(object):
    KEYS = [
        'observations',
        'states',
        'actions',
        'next_observations',
        'rewards',
        'dones',
        'infos',
    ]
    def __init__(self, num_envs, save_dir, success_only, max_traj):
        self.num_envs = num_envs
        self.save_dir = save_dir
        self.success_only = success_only
        self.traj = [
            {key: [] for key in self.KEYS} 
        for _ in range(num_envs)]
        self.data_to_save = [] 
        self.max_traj = max_traj

    def add_transition(self, obs, act, next_obs, rew, done, info):
        '''
            Important: All args need to be in format List[np.array()]
        '''
        for i in range(self.num_envs):
            if self.num_traj == self.max_traj:
                break
            self.traj[i]['observations'].append(obs[i])
            self.traj[i]['next_observations'].append(next_obs[i])
            self.traj[i]['actions'].append(act[i])
            self.traj[i]['rewards'].append(rew[i])
            self.traj[i]['dones'].append(done[i])
            self.traj[i]['infos'].append(info[i])
            if done[i]:
                if not self.success_only or info[i]['success']:
                    self.data_to_save.append(self.traj[i])
                    print(f"Trajectory number {self.num_traj} saved")
                self.traj[i] = {key: [] for key in self.KEYS}
    
    def add_state(self, s):
        for i in range(self.num_envs):
            self.traj[i]['states'].append(s[i])

    def add_obj_id(self, obj_id):
        assert self.num_envs == 1
        self.traj[0]['obj_id'] = obj_id

    def save(self, env_id=None):
        os.makedirs(self.save_dir, exist_ok=True)
        import pickle
        save_path = f'{self.save_dir}/{env_id}_trajectories_{len(self.data_to_save)}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(self.data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved {len(self.data_to_save)} trajectories to {save_path}')

    @property
    def num_traj(self):
        return len(self.data_to_save)