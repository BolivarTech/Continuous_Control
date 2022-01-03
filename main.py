"""
Continuos Control

By: Julian Bolivar
Version: 1.0.0
"""
from collections import deque

import numpy as np

from DDPG import ActorNet, CriticNet, Agent
from UnityEnv import UnityEnv

from argparse import ArgumentParser
import logging as log
import logging.handlers
import sys
import os

import csv
from datetime import datetime

# Main Logger
logHandler = None
logger = None
logLevel_ = logging.INFO
# OS running
OS_ = 'unknown'

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-t", "--train", required=False, action="store_true", help="Perform a model training, if -a "
                                                                                   "not specified a new model is "
                                                                                   "trained.")
    parser.add_argument("-p", "--play", required=False, action="store_true", help="Perform the model playing.")
    parser.add_argument("-a", "--actor", required=False, type=str, default=None,
                        help="Path to an pytorch actor model file.")
    parser.add_argument("-c", "--critic", required=False, type=str, default=None,
                        help="Path to an pytorch critic model file.")
    return parser


def save_scores(scores, computer_name):
    """

    :param scores:
    :return:
    """

    with open(f'{computer_name}-scores-{datetime.now().strftime("%Y%m%d%H%M%S")}.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(scores)


def train(actor_model_file, critic_model_file, computer_name):
    """
    Setup the training environment


    :param actor_model_file: file path with the actor model to be loaded
    :param critic_model_file: file path with the critic model to be loaded
    :param computer_name: String with the comnputer's name
    :return: None
    """

    global logHandler
    global logger
    global OS_
    global logLevel_

    u_env = None
    if OS_ == 'linux':
        u_env = UnityEnv(train_mode=True, env_filename="./SimEnv/Reacher_Linux/Reacher.x86_64", log_handler=logHandler)
    elif OS_ == 'win32':
        u_env = UnityEnv(train_mode=True, env_filename=".\\SimEnv\\Reacher_Windows_x86_64\\Reacher.exe", log_handler=logHandler)
    logger.info(f"Unity Environmet {OS_} loaded (001)")
    # number of agents in the environment
    logger.info(f'Number of agents: {u_env.get_num_agents()}')
    # number of actions
    logger.info(f'Number of actions: {u_env.get_num_actions()}')
    # examine the __state space
    logger.info(f'States look like: {u_env.get_state()}')
    logger.info(f'States have length: {u_env.get_state_size()}')
    # Generate the Actor Network
    actNet = ActorNet(u_env.get_state_size(), u_env.get_num_actions(), log_handler=logHandler)
    # Generate the Critic Network
    critNet = CriticNet(u_env.get_state_size(), u_env.get_num_actions(), 1, log_handler=logHandler)
    if actor_model_file is not None:
        actNet.load(actor_model_file)
    if critic_model_file is not None:
        critNet.load(critic_model_file)
    # train the agent
    agn = Agent(actormodel=actNet, criticmodel=critNet, log_handler=logHandler, agent_id=computer_name)  #, device='cpu')
    scores = agn.training(u_env)
    save_scores(scores,computer_name)
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

def play(actor_model_file, computer_name):
    """
    Perform an play using the agent

    :param actor_model_file: file path with the actor_model to be loaded
    :param computer_name: String with the comnputer's name
    """

    global logHandler
    global logger
    global OS_
    global logLevel_

    num_episodes = 100
    scores_window = deque(maxlen=num_episodes)  # last num_plays scores
    u_env = None
    if OS_ == 'linux':
        u_env = UnityEnv(train_mode=True, env_filename="./SimEnv/Reacher_Linux/Reacher.x86_64", log_handler=logHandler)
    elif OS_ == 'win32':
        u_env = UnityEnv(train_mode=True, env_filename=".\\SimEnv\\Reacher_Windows_x86_64\\Reacher.exe", log_handler=logHandler)
    logger.info(f"Unity Environmet {OS_} loaded (002)")
    # Generate the Actor Network
    actNet = ActorNet(u_env.get_state_size(), u_env.get_num_actions(), log_handler=logHandler, device='cpu')
    if actor_model_file is not None:
        actNet.load(actor_model_file)
    else:
        logger.error(f"Can't Play because a model file wasn't specified (003)")
        return
    actNet.eval()  # set policy network on eval mode
    actNet.set_grad_enabled(False)
    for i_episode in range(1, num_episodes + 1):
        state = u_env.reset(train_mode=False)
        score = u_env.get_score()
        t = 0
        while True:
            action = actNet.get_action(state)
            logger.debug(f"Episode: {i_episode} Step: {t} Action: {action} (004)")
            state, _, score, done = u_env.set_action(action)
            if np.any(done):
                score = np.squeeze(score,axis=(0, 1))
                logger.debug(f"Episode: {i_episode} DONE after {t} Steps (005)")
                logger.info(f"Episode: {i_episode} DONE with {score:.2f} score (006)")
                break
            t += 1
        scores_window.append(score)  # save most recent score
    mean_score = np.mean(scores_window)
    print(f"The mean score was {mean_score:.2f} over {num_episodes} episodes ")
    save_scores(scores_window,computer_name)

def main(computer_name):
    """
     Run the main function

    :param computer_name: String with the comnputer's name
    """

    global logger

    args = build_argparser().parse_args()
    actor_model_file = args.actor
    critic_model_file = args.critic
    if args.train and args.play:
        logger.error("Options Train and Play can't be used togethers (007)")
    elif args.train:
        if actor_model_file is not None:
            logger.debug(f"Training option selected with actor file {actor_model_file} (008)")
        else:
            logger.debug(f"Training option selected with new actor model (009)")
        if critic_model_file is not None:
            logger.debug(f"Training option selected with critic file {critic_model_file} (010)")
        else:
            logger.debug(f"Training option selected with new critic model (011)")
        train(actor_model_file, critic_model_file,computer_name)
    elif args.play:
        if critic_model_file is not None:
            logger.warning(f"On Play mode critic file {actor_model_file} is NOT needed (012)")
        if actor_model_file is not None:
            logger.debug(f"Play option selected with file {actor_model_file} (013)")
            play(actor_model_file,computer_name)
        else:
            logger.error(f"On Play option a actor model file must be specified (014)")
    else:
        logger.debug(f"Not option was selected with actor file {actor_model_file} (015)")
        logger.debug(f"Not option was selected with critic file {critic_model_file} (016)")


if __name__ == '__main__':

    computer_name = os.environ['COMPUTERNAME']
    loggPath = "."
    LogFileName = loggPath + '/' + computer_name +'-ccontrol.log'
    # Check where si running
    if sys.platform.startswith('freebsd'):
        OS_ = 'freebsd'
    elif sys.platform.startswith('linux'):
        OS_ = 'linux'
    elif sys.platform.startswith('win32'):
        OS_ = 'win32'
    elif sys.platform.startswith('cygwin'):
        OS_ = 'cygwin'
    elif sys.platform.startswith('darwin'):
        OS_ = 'darwin'
    if OS_ == 'linux':
        # loggPath = '/var/log/DNav'
        loggPath = './log'
        LogFileName = loggPath + '/' + computer_name +'-ccontrol.log'
    elif OS_ == 'win32':
        # loggPath = os.getenv('LOCALAPPDATA') + '\\DNav'
        loggPath = '.\\log'
        LogFileName = loggPath + '\\' + computer_name +'-ccontrol.log'

    # Configure the logger
    os.makedirs(loggPath, exist_ok=True)  # Create log path
    logger = log.getLogger('DCCtrl')  # Get Logger
    # Add the log message file handler to the logger
    logHandler = log.handlers.RotatingFileHandler(LogFileName, maxBytes=10485760, backupCount=5)
    # Logger Formater
    logFormatter = log.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                                 datefmt='%Y/%m/%d %H:%M:%S')
    logHandler.setFormatter(logFormatter)
    # Add handler to logger
    if 'logHandler' in globals():
        logger.addHandler(logHandler)
    else:
        logger.debug(f"logHandler NOT defined (017)")
        # Set Logger Lever
    # logger.setLevel(logging.INFO)
    logger.setLevel(logLevel_)
    # Start Running
    logger.debug(f"Running in {OS_} (018)")
    main(computer_name)
