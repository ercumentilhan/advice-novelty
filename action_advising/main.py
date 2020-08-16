import argparse
import gridworld.environment
import minatar_original
from action_advising.executor import Executor

from constants import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--process-index', type=int, default=0)
    parser.add_argument('--machine-name', type=str, default='HOME')

    parser.add_argument('--n-training-frames', type=int, default=2000000)
    parser.add_argument('--n-evaluation-trials', type=int, default=20)
    parser.add_argument('--evaluation-period', type=int, default=100)
    parser.add_argument('--evaluation-visualization-period', type=int, default=200)

    # ------------------------------------------------------------------------------------------------------------------
    # DQN
    parser.add_argument('--dqn-type', type=int, default=0)
    parser.add_argument('--dqn-gamma', type=float, default=0.99)
    parser.add_argument('--dqn-rm-type', type=str, default='uniform')
    parser.add_argument('--dqn-per-ims', action='store_true', default=True)
    parser.add_argument('--dqn-per-alpha', type=float, default=0.4)
    parser.add_argument('--dqn-per-beta', type=float, default=0.6)
    parser.add_argument('--dqn-rm-init', type=int, default=10000)
    parser.add_argument('--dqn-rm-max', type=int, default=100000)
    parser.add_argument('--dqn-target-update', type=int, default=1000)
    parser.add_argument('--dqn-batch-size', type=int, default=32)
    parser.add_argument('--dqn-learning-rate', type=float, default=0.0001)
    parser.add_argument('--dqn-train-per-step', type=int, default=1)
    parser.add_argument('--dqn-train-period', type=int, default=1)
    parser.add_argument('--dqn-adam-eps', type=float, default=0.00015)
    parser.add_argument('--dqn-eps-start', type=float, default=1.0)
    parser.add_argument('--dqn-eps-final', type=float, default=0.1)
    parser.add_argument('--dqn-eps-steps', type=int, default=100000)
    parser.add_argument('--dqn-huber-loss-delta', type=float, default=1.0)

    # ------------------------------------------------------------------------------------------------------------------
    # Action Advising
    parser.add_argument('--experiment-setup', type=int, default=0)
    parser.add_argument('--teacher-availability-step', type=int, default=0)
    parser.add_argument('--uc-threshold', type=float, default=0.1)
    parser.add_argument('--rnd-threshold', type=float, default=0.1)
    parser.add_argument('--rnd-learning-rate', type=float, default=0.0001)
    parser.add_argument('--rnd-compute-coeffs', action='store_true', default=False)
    parser.add_argument('--rnd-normalisation-steps', type=int, default=5000)

    # ------------------------------------------------------------------------------------------------------------------

    parser.add_argument('--use-gpu', action='store_true', default=False)
    parser.add_argument('--save-models', action='store_true', default=False)
    parser.add_argument('--visualization-period', type=int, default=100)
    parser.add_argument('--model-save-period', type=int, default=500)
    parser.add_argument('--env-name', type=str, default='')
    parser.add_argument('--env-training-seed', type=int, default=0)
    parser.add_argument('--env-evaluation-seed', type=int, default=1)
    parser.add_argument('--seed', type=int, default=100)

    # ------------------------------------------------------------------------------------------------------------------

    config = vars(parser.parse_args())

    env, eval_env = None, None
    if config['env_name'] == 'GridWorld':
        env = gridworld.environment.Environment(seed=config['env_training_seed'], slipping_prob=0.1)
        eval_env = gridworld.environment.Environment(seed=config['env_evaluation_seed'], slipping_prob=0.0)

    elif 'MinAtar' in config['env_name']:
        env_info = ENV_INFO[config['env_name']]
        env_name = env_info[4]
        difficulty_ramping = env_info[5]
        level = env_info[6]
        initial_difficulty = env_info[7]
        time_limit = env_info[8]

        if 'R' in env_info[0]:  # MinAtar Original
            env = minatar_original.Environment(env_name=env_name,
                                               sticky_action_prob=0.05,
                                               difficulty_ramping=True,
                                               random_seed=config['env_training_seed'],
                                               time_limit=time_limit + 1000)  # Timelimit is controlled by executor

            eval_env = minatar_original.Environment(env_name=env_name,
                                                    sticky_action_prob=0.0,
                                                    difficulty_ramping=True,
                                                    random_seed=config['env_evaluation_seed'],
                                                    time_limit=time_limit + 1000)

    executor = Executor(config, env, eval_env)
    executor.run()
