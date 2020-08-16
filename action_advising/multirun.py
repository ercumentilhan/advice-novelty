import os
import multiprocessing
import subprocess
import numpy as np
import shlex
import time
import math
import socket
from time import localtime, strftime
import hashlib

from constants import *

lock = multiprocessing.Lock()


def configToCommand(config):
    command = 'python main.py'
    for key in config[0]:
        command += ' --' + key + ' ' + str(config[0][key])
    for key in config[1]:
        if config[1][key]:
            command += ' --' + key
    return command


def stringToHash(string):
    return str(int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def work(cmd):
    lock.acquire()
    time.sleep(1)
    lock.release()
    return subprocess.call(shlex.split(cmd), shell=False)  # return subprocess.call(cmd, shell=False)


if __name__ == '__main__':

    machine_name = 'UNDEFINED'
    machine_id = None
    n_processors = 6

    hostname = socket.gethostname()

    if hostname == 'DESKTOP-LA8NF7N':  # HOME
        print('DESKTOP-LA8NF7 (HOME)')
        machine_name = 'HOME'
        n_processors = 9
        machine_id = 0
        config_set = 0
        n_seeds = 3
    elif hostname == 'DESKTOP-8A3QAR8':  # LAB-2
        print('DESKTOP-8A3QAR8 (LAB-2)')
        machine_name = 'LAB-2'
        n_processors = 4
        machine_id = 2
        config_set = 2
        n_seeds = 2
    elif hostname == 'DESKTOP-A90S73P':  # LAB-1
        print('DESKTOP-A90S73P (LAB-1)')
        machine_name = 'LAB-1'
        n_processors = 3
        machine_id = 1
        config_set = 1
        n_seeds = 4
    elif hostname == 'DESKTOP-DM48GMR':  # MSI Notebook
        print('DESKTOP-DM48GMR (MSI Notebook)')
        machine_name = 'MSI-NB'
        n_processors = 1
        machine_id = 3
        config_set = 3

    seeds_all = list(range((machine_id + 1) * 100, (machine_id + 1) * 100 + 20))

    # ------------------------------------------------------------------------------------------------------------------

    # Iterated hyperparameters
    experiment_setups_all = []
    dqn_type_s_all = []
    dqn_rm_type_s_all = []
    dqn_rm_s_all = []
    dqn_target_update_s_all = []
    dqn_epsilon_steps_s_all = []

    uc_size_s_all = []
    uc_prop_threshold_s_all = []
    uc_threshold_s_all = []

    rnd_threshold_s_all = []

    rm_regulation_threshold_s_all = []

    novelty_tabular_th_s_all = []

    # Static hyperparameters
    save_data = []
    n_training_frames = []
    n_evaluation_trials = []
    evaluation_period = []

    # Initialize config:
    config = [{}, {}]

    config[0]['run-id'] = 0
    config[0]['process-index'] = 0
    config[0]['machine-name'] = 'NONE'
    config[0]['n-training-frames'] = 2000000  # Frames
    config[0]['n-evaluation-trials'] = 20  # Episodes
    config[0]['evaluation-period'] = 100  # Frames

    config[0]['dqn-type'] = EGREEDY_DQN

    config[0]['dqn-type'] = 0
    config[0]['dqn-gamma'] = 0.99
    config[0]['dqn-rm-type'] = 'per'
    config[0]['dqn-rm-init'] = 10000
    config[0]['dqn-rm-max'] = 100000
    config[0]['dqn-per-alpha'] = 0.4
    config[0]['dqn-per-beta'] = 0.6
    config[1]['dqn-per-ims'] = True
    config[0]['dqn-target-update'] = 1000
    config[0]['dqn-batch-size'] = 32
    config[0]['dqn-learning-rate'] = 0.0001
    config[0]['dqn-train-per-step'] = 1
    config[0]['dqn-train-period'] = 2
    config[0]['dqn-adam-eps'] = 0.00015
    config[0]['dqn-eps-start'] = 1.0
    config[0]['dqn-eps-final'] = 0.05
    config[0]['dqn-eps-steps'] = 100000
    config[0]['dqn-huber-loss-delta'] = 1.0

    #config[1]['restore-expert'] = False

    config[0]['experiment-setup'] = 0

    config[1]['use-gpu'] = False
    config[1]['save-models'] = False
    config[0]['model-save-period'] = 200000  # Frames
    config[0]['visualization-period'] = 200 # Episodes
    config[0]['evaluation-visualization-period'] = 200  # Evaluation episodes
    config[0]['env-name'] = 'NONE'
    config[0]['env-training-seed'] = 0
    config[0]['env-evaluation-seed'] = 1
    config[0]['seed'] = 0


    config[0]['rnd-learning-rate'] = 0.0001


    config[0]['uc-threshold'] = 0.0
    config[0]['rnd-threshold'] = 0.0



    config[0]['teacher-availability-step'] = 0

    # Set to get RND coefficients (mean and std. dev.)
    config[1]['rnd-compute-coeffs'] = False
    config[0]['rnd-normalisation-steps'] = 10000  # GridWorld: 5000, MinAtar: 10000

    # ------------------------------------------------------------------------------------------------------------------

    # 0: 'none',
    # 1: 'early',
    # 2: 'random',
    # 3: 'uncertainty',
    # 4: 'adhoc_tabular',
    # 5: 'adhoc_rnd',
    # 6: 'novelty_tabular',
    # 7: 'novelty_rnd',
    # 8: 'novelty_uc_tabular',
    # 9: 'novelty_uc_rnd'

    n_seeds = 1
    n_processors = 2

    # env_names = ['GridWorld']
    env_names = ['GridWorld', 'MinAtar-asterix']

    #'MinAtar-asterix', 'MinAtar-breakout', 'MinAtar-freeway', 'MinAtar-seaquest', 'MinAtar-spaceinvaders'

    # 'MinAtar-breakout'# , , 'MinAtar-freeway', 'MinAtar-seaquest', 'MinAtar-spaceinvaders'

    # ------------------------------------------------------------------------------------------------------------------
    # GridWorld

    #experiment_setups_all.append([505])
    #            0: 'none',
    #        1: 'early',
    #        2: 'random',
    #        3: 'uncertainty',
    #        4: 'state_novelty',
    #        5: 'advice_novelty'
    #    }

    # 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001

    experiment_setups_all.append([104])  # budgets: 5, 8 - 305, 308, 405, 408, 505, 508

    dqn_type_s_all.append([NOISYNETS_DQN])
    dqn_rm_type_s_all.append(['uniform'])
    dqn_rm_s_all.append([(1000, 10000)])
    dqn_target_update_s_all.append([250])
    dqn_epsilon_steps_s_all.append([10000])

    uc_size_s_all.append([1000])
    uc_prop_threshold_s_all.append([0.5])

    uc_threshold_s_all.append([0.1])
    rnd_threshold_s_all.append([0.1])

    rm_regulation_threshold_s_all.append([0.01])

    save_data.append(False)
    n_training_frames.append(50000)
    n_evaluation_trials.append(5)
    evaluation_period.append(100)

    # ------------------------------------------------------------------------------------------------------------------

    # MinAtar
    experiment_setups_all.append([103])
    dqn_type_s_all.append([NOISYNETS_DQN])
    dqn_rm_type_s_all.append(['uniform'])
    dqn_rm_s_all.append([(10000, 100000)])
    dqn_target_update_s_all.append([1000])
    dqn_epsilon_steps_s_all.append([100000])

    uc_size_s_all.append([1000])
    uc_prop_threshold_s_all.append([0.5])

    uc_threshold_s_all.append([0.0])
    rnd_threshold_s_all.append([0.0])

    rm_regulation_threshold_s_all.append([0.01])

    save_data.append(False)
    n_training_frames.append(1500000)
    n_evaluation_trials.append(5)
    evaluation_period.append(1000)

    # ------------------------------------------------------------------------------------------------------------------

    i_parameter_set = 0
    i_command = 0
    commands = []

    seeds = seeds_all[:n_seeds]
    print('Seeds: ', seeds)

    for env_name in env_names:

        param_set_id = None
        if 'GridWorld' in env_name:
            param_set_id = 0
        elif 'MinAtar' in env_name:
            param_set_id = 1

        experiment_setups = experiment_setups_all[param_set_id]
        dqn_type_s = dqn_type_s_all[param_set_id]
        dqn_rm_type_s = dqn_rm_type_s_all[param_set_id]
        dqn_rm_s = dqn_rm_s_all[param_set_id]
        dqn_target_update_s = dqn_target_update_s_all[param_set_id]
        dqn_epsilon_steps_s = dqn_epsilon_steps_s_all[param_set_id]

        uc_threshold_s = uc_threshold_s_all[param_set_id]
        rnd_threshold_s = rnd_threshold_s_all[param_set_id]

        for experiment_setup in experiment_setups:
            for dqn_type in dqn_type_s:
                for dqn_rm_type in dqn_rm_type_s:
                    for dqn_rm in dqn_rm_s:
                        for dqn_target_update in dqn_target_update_s:
                            for dqn_epsilon_steps in dqn_epsilon_steps_s:
                                for uc_threshold in uc_threshold_s:
                                    for rnd_threshold in rnd_threshold_s:

                                        run_config = config.copy()
                                        run_config[0]['env-name'] = env_name
                                        run_config[0]['experiment-setup'] = experiment_setup
                                        run_config[0]['dqn-type'] = dqn_type
                                        run_config[0]['dqn-rm-type'] = dqn_rm_type
                                        run_config[0]['dqn-rm-init'] = dqn_rm[0]
                                        run_config[0]['dqn-rm-max'] = dqn_rm[1]
                                        run_config[0]['dqn-target-update'] = dqn_target_update
                                        run_config[0]['dqn-eps-steps'] = dqn_epsilon_steps

                                        if env_name == 'MinAtar-asterix':
                                            run_config[0]['uc-threshold'] = 0.0025  # Uncertainty
                                            if experiment_setup // 100 == 4:  # State Novelty
                                                run_config[0]['rnd-threshold'] = 0.1
                                            elif experiment_setup // 100 == 5:  # Advice Novelty
                                                run_config[0]['rnd-threshold'] = 0.025

                                        elif env_name == 'MinAtar-breakout':
                                            run_config[0]['uc-threshold'] = 0.0025  # Uncertainty
                                            if experiment_setup // 100 == 4:  # State Novelty
                                                run_config[0]['rnd-threshold'] = 0.1
                                            elif experiment_setup // 100 == 5:  # Advice Novelty
                                                run_config[0]['rnd-threshold'] = 0.025

                                        elif env_name == 'MinAtar-freeway':
                                            run_config[0]['uc-threshold'] = 0.0025  # Uncertainty
                                            if experiment_setup // 100 == 4:  # State Novelty
                                                run_config[0]['rnd-threshold'] = 0.1
                                            elif experiment_setup // 100 == 5:  # Advice Novelty
                                                run_config[0]['rnd-threshold'] = 0.025

                                        elif env_name == 'MinAtar-seaquest':
                                            run_config[0]['uc-threshold'] = 0.0025  # Uncertainty
                                            if experiment_setup // 100 == 4:  # State Novelty
                                                run_config[0]['rnd-threshold'] = 0.1
                                            elif experiment_setup // 100 == 5:  # Advice Novelty
                                                run_config[0]['rnd-threshold'] = 0.025

                                        elif env_name == 'MinAtar-spaceinvaders':
                                            run_config[0]['uc-threshold'] = 0.0025  # Uncertainty
                                            if experiment_setup // 100 == 4:  # State Novelty
                                                run_config[0]['rnd-threshold'] = 0.1
                                            elif experiment_setup // 100 == 5:  # Advice Novelty
                                                run_config[0]['rnd-threshold'] = 0.025

                                        elif env_name == 'GridWorld':
                                            run_config[0]['uc-threshold'] = 0.001  # Uncertainty
                                            if experiment_setup // 100 == 4:  # State Novelty
                                                run_config[0]['rnd-threshold'] = 0.0001
                                            elif experiment_setup // 100 == 5:  # Advice Novelty 0.00025
                                                run_config[0]['rnd-threshold'] = 0.001

                                        run_config[0]['n-training-frames'] = n_training_frames[param_set_id]
                                        run_config[0]['n-evaluation-trials'] = n_evaluation_trials[param_set_id]
                                        run_config[0]['evaluation-period'] = evaluation_period[param_set_id]

                                        run_id = str(ENV_INFO[run_config[0]['env-name']][0]) + '_' \
                                                 + str(DQN_TYPE_ABRV[run_config[0]['dqn-type']]) + '_' \
                                                 + str(experiment_setup).zfill(3) + '_' \
                                                 + str(i_parameter_set).zfill(3) + '_' \
                                                 + strftime("%Y%m%d-%H%M%S", localtime())

                                        run_config[0]['machine-name'] = str(machine_name)
                                        run_config[0]['process-index'] = str(i_command % n_processors)
                                        run_config[0]['run-id'] = str(run_id)

                                        for seed in seeds:
                                            seed_run_config = run_config.copy()
                                            seed_run_config[0]['seed'] = str(seed)
                                            commands.append(configToCommand(seed_run_config))
                                            i_command += 1
                                        i_parameter_set += 1

    # ==================================================================================================================

    print(commands)

    print('There are {} commands.'.format(len(commands)))

    n_cycles = int(math.ceil(len(commands) / n_processors))

    print('There are {} cycles.'.format(n_cycles))

    for i_cycle in range(n_cycles):
        pool = multiprocessing.Pool(processes=n_processors)

        start = (n_processors * i_cycle)
        end = start + n_processors

        print('start and end:', start, end)

        if end > len(commands):
            end = len(commands)

        print('start and end:', start, end)

        print(pool.map(work, commands[(n_processors * i_cycle):(n_processors * i_cycle) + n_processors]))
