import argparse
from environment import Env
import control as c
from model import DQN
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from collections import deque
import random
# import numpy as np
from tensorboardX import SummaryWriter
import os
import gc
import pathlib

# wandb
import wandb

# start a new wandb run to track this script
wandb.login()
run = wandb.init(
    project = "Baseline v2.2",

    config={
        "architecture": "Baseline"
    }
)
 
# Defining command-line arguments
parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--epochs', type=int, default=1000000, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--simnum', type=int, default=0, metavar='N')
parser.add_argument('--start-epoch', type=int, default=0, metavar='N')
parser.add_argument('--load-model', type=str, default='', metavar='N', help='load previous model')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--save-data', action='store_true', default=False)
parser.add_argument('--device', type=str, default="cpu", metavar='N')
parser.add_argument('--log-directory', type=str, default='', metavar='N', help='log directory')
parser.add_argument('--data-directory', type=str, default='', metavar='N', help='data directory')
# parser.add_argument('--history_size', type=int, default=4, metavar='N')
parser.add_argument('--history_size', type=int, default=1, metavar='N')
parser.add_argument('--width', type=int, default=129, metavar='N')
parser.add_argument('--height', type=int, default=84, metavar='N')
parser.add_argument('--hidden-size', type=int, default=32, metavar='N')
parser.add_argument('--epsilon', type=float, default=0.9, metavar='N')
parser.add_argument('--wepsilon', type=float, default=0.9, metavar='N')
parser.add_argument('--frame-time', type=float, default=0.2, metavar='N')
parser.add_argument('--reward', type=float, default=1, metavar='N')
parser.add_argument('--replay-size', type=int, default=3000, metavar='N')
args = parser.parse_args()
torch.manual_seed(args.seed)

if args.log_directory=='':
    args.log_directory = str(pathlib.Path(__file__).parent.absolute() / 'experiment/alphachu')+'/'
if args.data_directory=='':
    args.data_directory = str(pathlib.Path(__file__).parent.absolute() / 'data/alphachu')+'/'

# load the model with the lastest file timestamp name
if args.load_model=='':
    model_files = pathlib.Path(args.log_directory).glob('*/')
    latest_time = 0
    for model_file in model_files:
        # Extract timestamp from filename
        timestamp = model_file.stem.split('_')[0]
        if len(timestamp) == 12:  # Ensure it's a valid timestamp
            timestamp = int(timestamp)
            if latest_time is None or timestamp > latest_time:
                latest_time = timestamp
                args.load_model = str(model_file).split('/')[-1]


class Actor:
    def __init__(self):
        if args.device != 'cpu':
            torch.cuda.set_device(int(args.device))
            self.device = torch.device('cuda:{}'.format(int(args.device)))
        else:
            self.device = torch.device('cpu')

        self.simnum = args.simnum
        self.history_size = args.history_size
        self.height = args.height
        self.width = args.width
        self.hidden_size = args.hidden_size
        if args.test:
            args.epsilon = 0
            args.wepsilon = 0
        self.epsilon = args.epsilon
        self.log = args.log_directory + args.load_model + '/'
        self.writer = SummaryWriter(self.log + str(self.simnum) + '/')

        self.dis = 0.99 # discount factor
        self.win = False
        self.jump = False
        self.ground_key_dict = {0: c.stay,
                                1: c.left,
                                2: c.right,
                                3: c.up,
                                4: c.left_p,
                                5: c.right_p}
        self.jump_key_dict = {0: c.stay,
                              1: c.left_p,
                              2: c.right_p,
                              3: c.up_p,
                              4: c.p,
                              5: c.down_p}
        self.key_dict = self.ground_key_dict
        self.action_size = len(self.key_dict)
        self.replay_memory = deque(maxlen=args.replay_size)
        self.priority = deque(maxlen=args.replay_size)
        self.mainDQN = DQN(self.history_size, self.hidden_size, self.action_size).to(self.device)
        self.start_epoch = self.load_checkpoint()

    # Save cur state of actor
    def save_checkpoint(self, idx):
        checkpoint = {'simnum': self.simnum,
                      'epoch': idx + 1}
        torch.save(checkpoint, self.log + 'checkpoint{}.pt'.format(self.simnum))
        print('Actor {}: Checkpoint saved in '.format(self.simnum), self.log + 'checkpoint{}.pt'.format(self.simnum))

    def load_checkpoint(self):
        if os.path.isfile(self.log + 'checkpoint{}.pt'.format(self.simnum)):
            checkpoint = torch.load(self.log + 'checkpoint{}.pt'.format(self.simnum))
            self.simnum = checkpoint['simnum']
            print("Actor {}: loaded checkpoint ".format(self.simnum), '(epoch {})'.format(checkpoint['epoch']), self.log + 'checkpoint{}.pt'.format(self.simnum))
            return checkpoint['epoch']
        else: # start at default epoch
            print("Actor {}: no checkpoint found at ".format(self.simnum), self.log + 'checkpoint{}.pt'.format(self.simnum))
            return args.start_epoch

    # Saves memory and priority data to a file
    def save_memory(self):
        if os.path.isfile(self.log + 'memory.pt'):
            try:
                memory = torch.load(self.log + 'memory{}.pt'.format(self.simnum))
                memory['replay_memory'].extend(self.replay_memory)
                memory['priority'].extend(self.priority)
                torch.save(memory, self.log + 'memory{}.pt'.format(self.simnum))
                self.replay_memory.clear()
                self.priority.clear()
            except:
                time.sleep(10)
                memory = torch.load(self.log + 'memory{}.pt'.format(self.simnum))
                memory['replay_memory'].extend(self.replay_memory)
                memory['priority'].extend(self.priority)
                torch.save(memory, self.log + 'memory{}.pt'.format(self.simnum))
                self.replay_memory.clear()
                self.priority.clear()
        else:
            memory = {'replay_memory': self.replay_memory,
                      'priority': self.priority}
            torch.save(memory, self.log + 'memory{}.pt'.format(self.simnum))
            self.replay_memory.clear()
            self.priority.clear()

        print('Actor {}: Memory saved in '.format(self.simnum), self.log + 'memory{}.pt'.format(self.simnum))

    # loads pretrained model from file into mainDQN
    def load_model(self):
        if os.path.isfile(self.log + 'model.pt'):
            if args.device == 'cpu':
                model_dict = torch.load(self.log + 'model.pt', map_location=lambda storage, loc: storage)
            else:
                model_dict = torch.load(self.log + 'model.pt')
            self.mainDQN.load_state_dict(model_dict['state_dict'])
            print('Actor {}: Model loaded from '.format(self.simnum), self.log + 'model.pt')

        else:
            print("Actor {}: no model found at '{}'".format(self.simnum, self.log + 'model.pt'))

    # store last n states for TD error calc
    def history_init(self):
        history = torch.zeros([1, self.history_size, self.height, self.width])
        return history

    def update_history(self, history, state):
        history = torch.cat([state, history[:, :self.history_size - 1]], 1)
        return history

    def select_action(self, history):
        self.mainDQN.eval()
        history = history.to(self.device)
        qval = self.mainDQN(history) # get qval for each action
        self.maxv, action = torch.max(qval, 1) # get best action

        # generate random num to determine explore vs exploit
        sample = random.random()
        if not self.win:
            self.epsilon = args.epsilon
        else: # if winning is reached, use different strategy?
            self.epsilon = args.wepsilon
        if sample > self.epsilon: # exploit
            self.random = False
            action = action.item()
        else: # explore
            self.random = True
            action = random.randrange(self.action_size)
        return action

    # determine whether on ground or in air to execute action
    def control(self, jump):
        if not jump:
            self.key_dict = self.ground_key_dict
        elif jump:
            self.key_dict = self.jump_key_dict

    # interact with environment, learn from it, and log the progress over episodes
    def main(self):
        c.release()
        self.load_model()
        env.set_standard()
        total_reward = 0
        set_end = False


        for idx in range(self.start_epoch, args.epochs + 1):
            reward = self.round(idx, set_end) # run a round
            self.writer.add_scalar('reward', reward, idx) # log reward
            wandb.log({"reward": reward}, step=idx)
            total_reward += reward
            set_end = env.restart()

            if set_end: # game over
                self.writer.add_scalar('total_reward', total_reward, idx)
                wandb.log({"total_reward": total_reward}, step=idx)
                total_reward = 0
                self.win = False

                if not args.test: # save replay mem and model checkpoint, reload model
                    self.save_memory()
                    self.load_model()
                    self.save_checkpoint(idx)
                env.restart_set()
        self.writer.close()

    # execute a single point
    def round(self, round_num, set_end):
        print("Round {} Start".format(round_num))
        if not set_end:
            time.sleep(env.warmup)
        else:
            time.sleep(env.start_warmup)
        history = self.history_init()
        action = 0
        next_action = 0
        frame = 0
        reward = 0
        estimate = 0
        end = False
        maxv = torch.zeros(0).to(self.device)
        actions = torch.zeros(0).to(self.device)
        start_time = time.time()


        while not end:
            round_time = time.time() - start_time
            sleep_time = args.frame_time - (round_time % args.frame_time) # sync with frame rate
            time.sleep(sleep_time)
            start_time = time.time()
            if round_time + sleep_time > args.frame_time:
                raise ValueError('Timing error')
                # add while loop to shutdown pika and restart it

            # print(round_time, sleep_time, round_time + sleep_time)
            if args.save_data:
                save_dir = args.data_directory + str(args.time_stamp) + '-' + str(round_num) + '-' + str(frame) + '-' + str(action) + '.png'
            else:
                save_dir = None
            state = env.preprocess_img(save_dir=save_dir) # get current state
            next_history = self.update_history(history, state)
            end, jump = env.check_end()

            # if point is not over
            if not end:
                next_action = self.select_action(next_history) # select best action
                self.control(jump)
                print(self.key_dict[next_action]) # print command or key for next action
                self.key_dict[next_action](args.frame_time)

                if not self.random: # if exploiting
                    maxv = torch.cat([maxv, self.maxv])
                    actions = torch.cat([actions, torch.FloatTensor([action]).to(self.device)])
                frame += 1 # frame countere
                priority = abs(self.dis * self.maxv.item() - estimate)  # calculates pririty using TD error to determine importance
                estimate = self.maxv.item()
            else:
                c.release()
                if env.win:
                    reward = args.reward
                else: # -ve reward for losing
                    reward = - args.reward
                priority = abs(reward - estimate)
            
            if not args.test:
                self.replay_memory.append((history, action, reward, next_history, end))
                self.priority.append(priority)

            history = next_history # new state for next iteration
            action = next_action # update action for logging pirposes
            if frame > 2000:
                raise ValueError('Loop bug') # probably froze
        
        # to examine how predicted q value changes over time
        if maxv.size()[0] > 0:
            self.writer.add_scalar('maxv', maxv.mean(), round_num)
            wandb.log({"maxv": maxv.mean()})
        if actions.size()[0] > 0:
            self.writer.add_scalar('action', actions.mean(), round_num)
            wandb.log({"action": actions.mean()})
        self.writer.add_scalar('epsilon', self.epsilon, round_num)
        wandb.log({"epsilon": self.epsilon})
        self.writer.add_scalar('frame', frame, round_num)
        wandb.log({"frame": frame})
        gc.collect()
        if env.win:
            print("Round {} Win: reward:{}, frame:{}".format(round_num, reward, frame))
            self.win = True
        else:
            print("Round {} Lose: reward:{}, frame:{}".format(round_num, reward, frame))
            self.win = False
        return reward


if __name__ == "__main__":
    env = Env(args.height, args.width, args.frame_time)
    actor = Actor()
    actor.main()
