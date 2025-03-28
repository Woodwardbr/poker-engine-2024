import numpy as np
import time
import copy
from gymnasium import register


############################################
############################################

def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)['observation']

    # predicted
    ob = np.expand_dims(true_states[0],0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac,0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states

def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def mean_squared_error(a, b):
    return np.mean((a-b)**2)

############################################
############################################

def dict_obs_to_np_obs(dict_obs):
    obs_arr = []
    for o in dict_obs.values():
        if isinstance(o, int):
            obs_arr.append(o)
        elif isinstance(o, np.ndarray):
            obs_arr.extend(o.tolist())
    return np.array(obs_arr[1:-3]), dict_obs['is_my_turn']==1

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    # TODO: get this from hw1
    obs = env.reset()
    players = 1
    if players == 2:
        obs = list(obs[0])
        turn = [True,False]
        for i in range(2):
            obs[i], turn[i] = dict_obs_to_np_obs(obs[i])
    else:
        obs = obs[0]
        turn = True
        obs, turn = dict_obs_to_np_obs(obs)
    obses, acts, rews, nobses, terms, imgs = [], [], [], [], [], []
    steps = 0
    while True:
        if players == 2:
            for i in range(len(obs)):
                if turn[i]:
                    obses.append(obs[i])
                    act = policy.get_action(obs[i])
                    act = act.astype(int)
                    acts.append(act)
                    nobs, rew, done, _, mode = env.step(act)
                    for j in range(2):
                        obs[j], turn[j] =  dict_obs_to_np_obs(nobs[j])
                    nobses.append(obs[i])
                    rews.append(rew[i])
                    steps += 1

                    if done or steps > max_path_length:
                        terms.append(1)
                        break
                    else:
                        terms.append(0)
        else:
            if turn:
                obses.append(obs)
                act = policy.get_action(obs)
                act = act.astype(int)
                acts.append(act)
                nobs, rew, done, _, mode = env.step(act)
                for j in range(2):
                    obs, turn =  dict_obs_to_np_obs(nobs)
                nobses.append(obs)
                rews.append(rew)
                steps += 1

                if done or steps > max_path_length:
                    terms.append(1)
                    break
                else:
                    terms.append(0)

        if done:
            break
    return Path(obses, imgs, acts, rews, nobses, terms)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    # TODO: get this from hw1
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)
        print('sampled {}/{} timesteps'.format(timesteps_this_batch, min_timesteps_per_batch), end='\r')

    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    # TODO: get this from hw1
    paths = []
    for i in range(ntraj):
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
        print('sampled {}/ {} trajs'.format(i, ntraj), end='\r')
    return paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data

def register_custom_env():
    register(
        id='PokerEnv',
        entry_point='engine.gym_env:PokerEnv'
    )