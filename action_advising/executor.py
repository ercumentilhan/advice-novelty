import os
import psutil
from time import localtime, strftime
import pathlib
import glob
import shutil
import random
import numpy as np
import copy
import math
from collections import deque
import pickle

import cv2
import tensorflow as tf

from dqn_egreedy import EpsilonGreedyDQN
from dqn_noisynets import NoisyNetsDQN


from statistics import Statistics
from rnd import RND

cv2.ocl.setUseOpenCL(False)
os.environ['TF_CPP_MIN_LONG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from constants import *

plt.rcParams.update({'font.size': 14})


class Executor:
    def __init__(self, config, env, eval_env):
        self.config = config
        self.env = env
        self.eval_env = eval_env

        self.stats = None
        self.teacher_stats = None

        self.actor_agent = None
        self.actor_agent_rnd = None

        self.evaluation_dir = None
        self.save_videos_path = None

        self.steps_reward = 0.0
        self.steps_error_in = 0.0
        self.steps_error_out = 0.0

        self.episode_duration = 0
        self.episode_reward = 0.0
        self.episode_error_in = 0.0
        self.episode_error_out = 0.0

        self.process = None
        self.run_id = None

        self.scripts_dir = None
        self.local_workspace_dir = None

        self.runs_local_dir = None
        self.summaries_dir = None
        self.checkpoints_dir = None
        self.copy_scripts_dir = None
        self.videos_dir = None

        self.save_summary_path = None
        self.save_model_path = None
        self.save_scripts_path = None
        self.save_videos_path = None

        self.session = None
        self.summary_writer = None
        self.saver = None

        self.teacher_agent = None

        # Action advising
        self.action_advising_enabled = None
        self.action_advising_budget = None
        self.action_advising_method = None

        # Rendering
        self.obs_images = []

        # RND observation normalization
        self.obs_running_mean = None
        self.obs_running_std = None
        self.obs_norm_n = 0
        self.obs_norm_max_n = None  # Determined later

    # ------------------------------------------------------------------------------------------------------------------

    def render(self, env):
        if self.config['env_type'] == GRIDWORLD:
            return env.render()
        elif self.config['env_type'] == MINATAR:
            return env.render_state()

    # ------------------------------------------------------------------------------------------------------------------

    def run(self):
        self.process = psutil.Process(os.getpid())

        os.environ['PYTHONHASHSEED'] = str(self.config['seed'])
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        tf.set_random_seed(self.config['seed'])

        self.run_id = self.config['run_id']
        self.seed_id = str(self.config['seed'])

        print('Run ID: {}'.format(self.run_id))

        # --------------------------------------------------------------------------------------------------------------

        self.scripts_dir = os.path.dirname(os.path.abspath(__file__))
        self.local_workspace_dir = os.path.join(str(pathlib.Path(self.scripts_dir).parent.parent.parent.parent))

        print('{} (Scripts directory)'.format(self.scripts_dir))
        print('{} (Local Workspace directory)'.format(self.local_workspace_dir))

        self.runs_local_dir = os.path.join(self.local_workspace_dir, 'Runs-AA')
        os.makedirs(self.runs_local_dir, exist_ok=True)

        self.summaries_dir = os.path.join(self.runs_local_dir, 'Summaries')
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.runs_local_dir, 'Checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.copy_scripts_dir = os.path.join(self.runs_local_dir, 'Scripts')
        os.makedirs(self.copy_scripts_dir, exist_ok=True)

        self.videos_dir = os.path.join(self.runs_local_dir, 'Videos')
        os.makedirs(self.videos_dir, exist_ok=True)

        self.data_dir = os.path.join(self.runs_local_dir, 'Data')
        os.makedirs(self.data_dir, exist_ok=True)

        self.replay_memory_dir = os.path.join(self.runs_local_dir, 'ReplayMemory')
        os.makedirs(self.replay_memory_dir, exist_ok=True)

        self.save_summary_path = os.path.join(self.summaries_dir, self.run_id, self.seed_id)
        self.save_model_path = os.path.join(self.checkpoints_dir, self.run_id, self.seed_id)
        self.save_scripts_path = os.path.join(self.copy_scripts_dir, self.run_id, self.seed_id)
        self.save_videos_path = os.path.join(self.videos_dir, self.run_id, self.seed_id)
        self.save_data_path = os.path.join(self.data_dir, self.run_id, self.seed_id)
        self.save_replay_memory_path = os.path.join(self.replay_memory_dir, self.run_id, self.seed_id)

        if self.config['save_models']:
            os.makedirs(self.save_model_path, exist_ok=True)

        os.makedirs(self.save_videos_path, exist_ok=True)
        os.makedirs(self.save_data_path, exist_ok=True)

        self.copy_scripts(self.save_scripts_path)

        # --------------------------------------------------------------------------------------------------------------

        if self.config['use_gpu']:
            print('Using GPU.')
            session_config = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1)
        else:
            print('Using CPU.')
            session_config = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1,
                allow_soft_placement=True,
                device_count={'CPU': 1, 'GPU': 0})

        self.session = tf.InteractiveSession(graph=tf.get_default_graph(), config=session_config)
        self.summary_writer = tf.summary.FileWriter(self.save_summary_path, self.session.graph)

        self.stats = Statistics(self.summary_writer, self.session)
        self.teacher_stats = Statistics(self.summary_writer, self.session)

        # --------------------------------------------------------------------------------------------------------------

        self.env_info = {}

        env_info = ENV_INFO[self.config['env_name']]
        env_shortname = env_info[4]
        difficulty_ramping = env_info[5]
        level = env_info[6]
        initial_difficulty = env_info[7]
        self.env_info['max_timesteps'] = env_info[8]

        self.config['env_type'] = env_info[1]
        self.config['env_obs_form'] = env_info[2]
        self.config['env_states_are_countable'] = env_info[3]

        if self.config['env_type'] == GRIDWORLD:
            self.config['env_obs_dims'] = self.env.obs_space.shape
            self.config['env_n_actions'] = self.env.action_space.n
        elif self.config['env_type'] == MINATAR:
            self.config['env_obs_dims'] = self.env.state_shape()
            self.config['env_n_actions'] = self.env.num_actions()

        # --------------------------------------------------------------------------------------------------------------

        self.apply_experiment_setup()

        # --------------------------------------------------------------------------------------------------------------
        # Setup actor agent
        self.config['actor_id'] = self.run_id

        if self.config['dqn_type'] == EGREEDY_DQN:
            self.actor_agent = EpsilonGreedyDQN(self.config['actor_id'], self.config, self.session,
                                                self.config['dqn_eps_start'],
                                                self.config['dqn_eps_final'],
                                                self.config['dqn_eps_steps'], self.stats)
        elif self.config['dqn_type'] == NOISYNETS_DQN:
            self.actor_agent = NoisyNetsDQN(self.config['actor_id'], self.config, self.session, self.stats)


        self.config['actor_id'] = self.actor_agent.id

        print('Actor ID: {}'.format(self.actor_agent.id))

        # --------------------------------------------------------------------------------------------------------------
        # Setup actor agent's RND if needed
        if 'novelty' in self.action_advising_method:
            self.actor_agent_rnd = RND(self.config['actor_id'], self.config, self.session,
                                       self.config['rnd_learning_rate'])

            if self.action_advising_method == 'state_novelty':
                self.actor_agent.rnd_model = self.actor_agent_rnd

        # --------------------------------------------------------------------------------------------------------------

        self.save_config(self.config, os.path.join(self.save_summary_path, 'config.txt'))

        # --------------------------------------------------------------------------------------------------------------
        # Initialise the teacher agent
        if self.action_advising_method != 'none' and self.config['env_type'] == MINATAR:
            expert_info = EXPERT[self.config['env_name']]
            self.config['teacher_id'] = expert_info[0]
            if 'EG' in self.config['teacher_id']:
                self.teacher_agent = EpsilonGreedyDQN(self.config['teacher_id'], self.config, self.session, 0.0, 0.0, 1,
                                                  self.stats)
            elif 'NN' in self.config['teacher_id']:
                self.teacher_agent = NoisyNetsDQN(self.config['teacher_id'], self.config, self.session,
                                                  self.stats)
            self.teacher_agent.restore(self.checkpoints_dir, expert_info[0] + '/' + expert_info[1], expert_info[2])

        # --------------------------------------------------------------------------------------------------------------

        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Number of parameters: {}'.format(total_parameters))

        self.saver = tf.train.Saver(max_to_keep=None)
        self.session.run(tf.global_variables_initializer())

        # --------------------------------------------------------------------------------------------------------------
        # Restore the teacher agent
        if self.action_advising_method != 'none' and self.config['env_type'] == MINATAR:
            expert_info = EXPERT[self.config['env_name']]
            self.teacher_agent.restore(self.checkpoints_dir, expert_info[0] + '/' + expert_info[1], expert_info[2])

        # --------------------------------------------------------------------------------------------------------------

        if not self.config['save_models']:
            tf.get_default_graph().finalize()

        reward_is_seen = False

        eval_score = self.evaluate()
        print('Evaluation @ {} | {}'.format(self.stats.n_env_steps, eval_score))

        obs, render = self.reset_env()

        while True:
            # ----------------------------------------------------------------------------------------------------------
            # RND observation normalisation
            if self.config['rnd_compute_coeffs']:
                if self.obs_norm_n < self.config['rnd_normalisation_steps']:
                    obs_mean = obs.mean(axis=(0, 1))
                    obs_std = obs.std(axis=(0, 1))
                    if self.obs_norm_n == 0:
                        self.obs_running_mean = obs_mean
                        self.obs_running_std = obs_std
                    else:
                        self.obs_running_mean = \
                            self.obs_running_mean + (obs_mean - self.obs_running_mean) / (self.obs_norm_n + 1)
                        self.obs_running_std = \
                            self.obs_running_std + (obs_std - self.obs_running_std) / (self.obs_norm_n + 1)
                    self.obs_norm_n += 1

                if self.obs_norm_n == self.config['rnd_normalisation_steps']:
                    print(repr(self.obs_running_mean))
                    print(repr(self.obs_running_std))
                    self.obs_norm_n += 1
                    import sys
                    sys.exit()

            # ----------------------------------------------------------------------------------------------------------
            # Determine state id
            state_id = None
            if self.config['env_states_are_countable']:
                state_id = self.env.get_state_id()

            # ----------------------------------------------------------------------------------------------------------
            # Determine action

            # Action advising
            get_action_advice = False
            if self.action_advising_method != 'none' and \
                    self.action_advising_budget > 0 and \
                    self.stats.n_env_steps >= self.config['teacher_availability_step']:

                if self.action_advising_method == 'early':
                    get_action_advice = True

                elif self.action_advising_method == 'random':
                    if random.random() < 0.5:
                        get_action_advice = True

                elif self.action_advising_method == 'uncertainty':
                    uncertainty = self.actor_agent.get_uncertainty(obs, True)
                    probability = np.clip(uncertainty / (2 * self.config['uc_threshold']), 0.0, 1.0)
                    if probability >= random.random():
                        get_action_advice = True

                elif self.action_advising_method == 'state_novelty':
                    rnd_value = self.actor_agent_rnd.get_error(obs, evaluation=False, normalize=True)
                    probability = np.clip(rnd_value / (2 * self.config['rnd_threshold']), 0.0, 1.0)
                    if probability >= random.random():
                        get_action_advice = True

                elif self.action_advising_method == 'advice_novelty':
                    rnd_value = self.actor_agent_rnd.get_error(obs, evaluation=False, normalize=True)
                    probability = np.clip(rnd_value/(2 * self.config['rnd_threshold']), 0.0, 1.0)
                    if probability >= random.random():
                        get_action_advice = True

            if get_action_advice:
                self.action_advising_budget -= 1
                self.stats.advices_taken += 1
                self.stats.advices_taken_cumulative += 1

                # Budget is 0, detach RND model to prevent further updates for computational efficiency
                if self.action_advising_method == 'state_novelty' and self.action_advising_budget == 0:
                    self.actor_agent.rnd_model = None

                # Teacher action
                if self.config['env_type'] == GRIDWORLD:
                    action = self.env.optimal_action()
                elif self.config['env_type'] == MINATAR:
                    action = self.teacher_agent.get_greedy_action(obs)

                source = 1
            else:
                action = self.actor_agent.get_action(obs)
                source = 0

            # ----------------------------------------------------------------------------------------------------------
            # Determine transition identifier
            transition_id = None
            if self.config['env_states_are_countable']:
                transition_id = self.env.get_transition_id(action)

            if get_action_advice and self.action_advising_method == 'advice_novelty':
                self.actor_agent_rnd.train_model(obs, loss_id=0, is_batch=False, normalize=True)

            # ----------------------------------------------------------------------------------------------------------
            # Execute action

            obs_next, reward, done = None, None, None
            if self.config['env_type'] == GRIDWORLD:
                obs_next, reward, done = self.env.step(action)
            elif self.config['env_type'] == MINATAR:
                reward, done = self.env.act(action)
                obs_next = self.env.state().astype(dtype=np.float32)

            if render:
                self.obs_images.append(self.render(self.env))

            self.episode_reward += reward
            self.episode_duration += 1

            self.steps_reward += reward
            self.stats.n_env_steps += 1

            if reward > 0 and reward_is_seen is False:
                reward_is_seen = True
                print(">>> Reward is seen at ", self.stats.n_episodes, "|", self.episode_duration)

            # ----------------------------------------------------------------------------------------------------------
            # Feedback
            old_transition = self.actor_agent.feedback_observe(obs, action, reward, obs_next, done,
                                                               source=source,
                                                               state_id=state_id,
                                                               transition_id=transition_id)

            # ----------------------------------------------------------------------------------------------------------

            td_error_batch, loss = self.actor_agent.feedback_learn()
            td_error_batch_sum = np.sum(td_error_batch)


            self.episode_error_out += td_error_batch_sum
            self.steps_error_out += td_error_batch_sum

            self.stats.loss += loss
            obs = obs_next
            done = done or self.episode_duration >= self.env_info['max_timesteps']

            if done:
                self.stats.n_episodes += 1
                self.stats.episode_reward_auc += np.trapz([self.stats.episode_reward_last, self.episode_reward])
                self.stats.episode_reward_last = self.episode_reward

                self.stats.update_summary_episode(self.episode_reward, self.stats.episode_reward_auc,
                                                  self.episode_duration, self.episode_error_in, self.episode_error_out)

                print('{}'.format(self.stats.n_episodes), end=' | ')
                print('{:.1f}'.format(self.episode_reward), end=' | ')
                print('{}'.format(self.episode_duration), end=' | ')
                print('{}'.format(self.stats.n_env_steps))

                if render:
                    self.write_video(self.obs_images, '{}_{}'.format(str(self.stats.n_episodes - 1),
                                                                     str(self.stats.n_env_steps - self.episode_duration)))

                obs, render = self.reset_env()

            # Per N steps summary update
            if self.stats.n_env_steps % self.stats.n_steps_per_update == 0:
                self.stats.steps_reward_auc += np.trapz([self.stats.steps_reward_last, self.steps_reward])
                self.stats.steps_reward_last = self.steps_reward
                self.stats.epsilon = self.actor_agent.eps if self.actor_agent.type == EGREEDY_DQN else 0

                self.stats.update_summary_steps(self.steps_reward, self.stats.steps_reward_auc,
                                                self.steps_error_in, self.steps_error_out)

                self.stats.advices_taken = 0.0
                self.stats.exploration_steps_taken = 0
                self.steps_reward = 0.0
                self.steps_error_in = 0.0
                self.steps_error_out = 0.0

            if self.stats.n_env_steps % self.config['evaluation_period'] == 0:
                evaluation_score = self.evaluate()
                print('Evaluation ({}): {}'.format(self.stats.n_episodes, evaluation_score))

            if self.config['save_models'] and self.stats.n_env_steps % self.config['model_save_period'] == 0:
                model_path = os.path.join(os.path.join(self.save_model_path), 'model-{}.ckpt').format(
                    self.stats.n_env_steps)
                print('[{}] Saving model... {}'.format(self.stats.n_env_steps, model_path))
                self.saver.save(self.session, model_path)

            if self.stats.n_env_steps >= self.config['n_training_frames']:
                if self.config['save_models']:
                    model_path = os.path.join(os.path.join(self.save_model_path), 'model-{}.ckpt').format(
                        self.stats.n_env_steps)
                    print('[{}] Saving model... {}'.format(self.stats.n_env_steps, model_path))
                    self.saver.save(self.session, model_path)
                break

        print('Env steps: {}'.format(self.stats.n_env_steps))

        self.session.close()

    # ------------------------------------------------------------------------------------------------------------------

    def reset_env(self):

        self.episode_duration = 0
        self.episode_reward = 0.0
        self.episode_error_in = 0.0
        self.episode_error_out = 0.0

        render = self.stats.n_episodes % self.config['visualization_period'] == 0

        if render:
            self.obs_images.clear()

        obs = None
        if self.config['env_type'] == GRIDWORLD:
            obs = self.env.reset()
        elif self.config['env_type'] == MINATAR:
            self.env.reset()
            obs = self.env.state().astype(dtype=np.float32)

        if render:
            self.obs_images.append(self.render(self.env))
        return obs, render

    # ------------------------------------------------------------------------------------------------------------------

    def write_video(self, images, filename):
        v_w = np.shape(images[0])[0]
        v_h = np.shape(images[0])[1]
        filename_full = os.path.join(self.save_videos_path, str(filename))
        video = cv2.VideoWriter(filename_full + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (v_h, v_w))
        for image in images:
            video.write(image)
        video.release()

    def copy_scripts(self, target_directory):
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        files = glob.iglob(os.path.join(self.scripts_dir, '*.py'))
        for file in files:
            if os.path.isfile(file):
                shutil.copy2(file, target_directory)

    def print_memory_usage(self):
        print('-- RAM: {}'.format(self.process.memory_info().rss / (1024 * 1024)))

    def save_config(self, config, filepath):
        fo = open(filepath, "w")
        for k, v in config.items():
            fo.write(str(k) + '>> ' + str(v) + '\n')
        fo.close()

    # ------------------------------------------------------------------------------------------------------------------

    def evaluate(self):
        eval_render = self.stats.n_evaluations % self.config['evaluation_visualization_period'] == 0

        eval_total_reward = 0.0
        eval_duration = 0

        self.eval_env.set_random_state(self.config['env_evaluation_seed'])

        for i_eval_trial in range(self.config['n_evaluation_trials']):
            eval_obs_images = []

            eval_obs = None
            if self.config['env_type'] == GRIDWORLD:
                eval_obs = self.eval_env.reset()
            elif self.config['env_type'] == MINATAR:
                self.eval_env.reset()
                eval_obs = self.eval_env.state().astype(dtype=np.float32)

            eval_episode_reward = 0.0
            eval_episode_duration = 0

            while True:
                if eval_render:
                    eval_obs_images.append(self.render(self.eval_env))

                eval_action = self.actor_agent.get_greedy_action(eval_obs)

                eval_obs_next, eval_reward, eval_done = None, None, None
                if self.config['env_type'] == GRIDWORLD:
                    eval_obs_next, eval_reward, eval_done = self.eval_env.step(eval_action)
                elif self.config['env_type'] == MINATAR:
                    eval_reward, eval_done = self.eval_env.act(eval_action)
                    eval_obs_next = self.eval_env.state().astype(dtype=np.float32)

                eval_episode_reward += eval_reward
                eval_duration += 1
                eval_episode_duration += 1
                eval_obs = eval_obs_next

                eval_done = eval_done or eval_episode_duration >= self.env_info['max_timesteps']

                if eval_done:

                    # Gridworld - score correction
                    if self.config['env_type'] == GRIDWORLD:
                        if eval_episode_reward == 0:
                            if self.eval_env.state.agent_dead:
                                pos = self.eval_env.state.agent_pos_prev
                                eval_episode_reward = 23 - self.eval_env.get_pos_dist_to_goal(0, pos[0], pos[1])
                            else:
                                pos = self.eval_env.state.agent_pos
                                eval_episode_reward = 23 - self.eval_env.get_pos_dist_to_goal(0, pos[0], pos[1])
                        elif eval_episode_reward == 1:
                            eval_episode_reward = 23 + (self.env_info['max_timesteps'] - eval_episode_duration)

                        eval_episode_reward /= (self.env_info['max_timesteps'] + 1)

                    if eval_render:
                        eval_obs_images.append(self.render(self.eval_env))
                        self.write_video(eval_obs_images, 'E_{}_{}'.format(str(self.stats.n_episodes),
                                                                           str(self.stats.n_env_steps)))
                        eval_obs_images.clear()
                        eval_render = False
                    eval_total_reward += eval_episode_reward

                    break

        eval_mean_reward = eval_total_reward / float(self.config['n_evaluation_trials'])

        self.stats.evaluation_reward_auc += np.trapz([self.stats.evaluation_reward_last, eval_mean_reward])
        self.stats.evaluation_reward_last = eval_mean_reward

        self.stats.n_evaluations += 1

        self.stats.update_summary_evaluation(eval_mean_reward, eval_duration, self.stats.evaluation_reward_auc)

        return eval_mean_reward

    # ------------------------------------------------------------------------------------------------------------------

    # Experiment setups (abc):
    # 0: No Advising
    #
    # a: Action advising method
    # -- 0: None
    # -- 1: Early advising
    # -- 2: Uniformly random advising
    #
    # c: Budget

    def apply_experiment_setup(self):

        action_advising_methods = {
            0: 'none',
            1: 'early',
            2: 'random',
            3: 'uncertainty',
            4: 'state_novelty',
            5: 'advice_novelty'
        }

        env_code = ENV_INFO[self.config['env_name']][0]

        if 'GW' in env_code:
            action_advising_budgets = {
                0: 100,
                1: 250,
                2: 500,
                3: 1000,
                4: 2500,
                5: 5000,
                6: 10000,
                7: 25000,
                8: 50000,
                9: 100000
            }
        elif 'MA'in env_code:
            action_advising_budgets = {
                0: 1000,
                1: 2500,
                2: 5000,
                3: 10000,
                4: 25000,
                5: 50000,
                6: 100000,
                7: 250000,
                8: 500000,
                9: 1000000
            }

        self.action_advising_method = 'none'
        self.rm_regulation_method = 'none'
        self.action_advising_budget = 0

        if self.config['experiment_setup'] != 0:
            param_1 = self.config['experiment_setup'] // 100
            param_3 = ((self.config['experiment_setup'] % 100) % 10)

            self.action_advising_method = action_advising_methods[param_1]
            self.action_advising_budget = action_advising_budgets[param_3]

        print('Experiment:', self.action_advising_method, self.action_advising_budget)

    # ------------------------------------------------------------------------------------------------------------------
