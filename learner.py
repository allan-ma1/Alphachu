# To execute:
# python learner.py --load-model 240415125014_128_0.0001_4_84_129_32_1_30000_1500_10

import argparse
import control as c
from model import DQN
import datetime
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import torch.cuda
import torch.backends.cudnn as cudnn
from collections import deque
from tensorboardX import SummaryWriter
import numpy as np
import os
import gc
import pathlib
cudnn.benchmark = True

# wandb
import wandb

# start a new wandb run to track this script
wandb.login()
run = wandb.init(
    project = "Dynamic 8-4 v1",

    config={
        "architecture": "Baseline"
    }
)
 
# Defining command-line arguments
parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 256)') # changed to 128
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--gpu', type=int, default=0, metavar='N',
                    help='number of cuda')
parser.add_argument('--no-cuda', action='store_true', default=False,
                                        help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--time-stamp', type=str, default=datetime.datetime.now().strftime("%y%m%d%H%M%S"), metavar='N',
                    help='time of the run(no modify)')
parser.add_argument('--load-model', type=str, default='000000000000', metavar='N',
                    help='load previous model')
# eg: --load-model 240403031024_128_0.0001_4_84_129_32_1_30000_1500_10
parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                    help='start-epoch number')
parser.add_argument('--log-directory', type=str, default='', metavar='N',
                    help='log directory')
# parser.add_argument('--history_size', type=int, default=4, metavar='N')
parser.add_argument('--history_size', type=int, default=8, metavar='N')
parser.add_argument('--width', type=int, default=129, metavar='N')
parser.add_argument('--height', type=int, default=84, metavar='N')
parser.add_argument('--hidden-size', type=int, default=32, metavar='N')
parser.add_argument('--action-size', type=int, default=6, metavar='N')
parser.add_argument('--reward', type=int, default=1, metavar='N')
parser.add_argument('--replay-size', type=int, default=30000, metavar='N')
parser.add_argument('--update-cycle', type=int, default=1500, metavar='N')
parser.add_argument('--actor-num', type=int, default=10, metavar='N')
args = parser.parse_args()
torch.cuda.set_device(args.gpu)
args.device = torch.device("cuda:{}".format(args.gpu) if not args.no_cuda and torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)

config_list = [args.batch_size, args.lr, args.history_size,
               args.height, args.width, args.hidden_size,
               args.reward, args.replay_size,
               args.update_cycle, args.actor_num]
config = ""
for i in map(str, config_list):
    config = config + '_' + i
print("Config:", config)

# default properly gets the path --> /home/chay/Allan/ape-x-pytorch/experiment/alphachu/
if args.log_directory=='':
    args.log_directory = str(pathlib.Path(__file__).parent.absolute() / 'experiment/alphachu')+'/'
    

class Learner():
    def __init__(self):
        # Initializes parameters
        self.device = args.device
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.history_size = args.history_size
        self.replay_size = args.replay_size
        self.width = args.width     # height and width of image
        self.height = args.height
        self.hidden_size = args.hidden_size
        self.action_size = args.action_size
        self.update_cycle = args.update_cycle
        self.log_interval = args.log_interval
        self.actor_num = args.actor_num 
        self.alpha = 0.7 # degree of prioritization
        self.beta_init = 0.4 # importance sampling weights
        self.beta = self.beta_init
        self.beta_increment = 1e-6
        self.e = 1e-6
        self.dis = 0.99 # discount factor
        self.start_epoch = 0

        # Primary model for learning and decision making --> approxiamates Q function
        self.mainDQN = DQN(self.history_size, self.hidden_size, self.action_size).to(self.device)

        # Generate Q values that the outputs of mainDQN are compared against --> stabilizes learning
        #  Updated periodically --> copied over from mainDQN, updated asynchronously
        self.targetDQN = DQN(self.history_size, self.hidden_size, self.action_size).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.mainDQN.parameters(), lr=args.lr)
        self.replay_memory = deque(maxlen=self.replay_size) # double ended queue to store experiences
        self.priority = deque(maxlen=self.replay_size) # store priorities

        # previous model declared --> pull from that directory
        if args.load_model != '000000000000':
            # print("LOG DIR:", args.log_directory)
            self.log = args.log_directory + args.load_model + '/'
            args.time_stamp = args.load_model[:12]
            args.start_epoch = self.load_model()
        # if no model declared --> uses default which "" (log dir) + cur_timestamp + ... 
        # print("SELF.LOG before:", self.log)
        # config stores the hyperparameters --> will save to a different path if hyperparams changed
        self.log = args.log_directory + args.time_stamp + config + '/'
        # print("SELF.LOG after :", self.log)
        self.writer = SummaryWriter(self.log)

        # for the dynamic steps algorithm
        self.start_channels = 8
        self.min_channels = 4
        self.reduction_step = 15000

    # Updates targetDQN asynchronously --> copies values from mainDQN
    def update_target_model(self):
        self.targetDQN.load_state_dict(self.mainDQN.state_dict())

    def save_model(self, train_epoch):
        model_dict = {'state_dict': self.mainDQN.state_dict(),
                      'optimizer_dict': self.optimizer.state_dict(),
                      'train_epoch': train_epoch}
        torch.save(model_dict, self.log + 'model.pt')
        print('Learner: Model saved in ', self.log + 'model.pt')

    def load_model(self):
        print("SELF.LOG", self.log) # --> the timestamp folder path
        # checks if file model.pt exists in self.log
        if os.path.isfile(self.log + 'model.pt'):
            model_dict = torch.load(self.log + 'model.pt')
            self.mainDQN.load_state_dict(model_dict['state_dict'])
            self.optimizer.load_state_dict(model_dict['optimizer_dict'])
            self.update_target_model()
            self.start_epoch = model_dict['train_epoch']
            print("Learner: Model loaded from {}(epoch:{})".format(self.log + 'model.pt', str(self.start_epoch)))
        else:
            raise "=> Learner: no model found at '{}'".format(self.log + 'model.pt')

    # Load saved replay memory and priority values into Leanrer's replay mem and priority queue
    def load_memory(self, simnum):
        # check if file exists at that path
        if os.path.isfile(self.log + 'memory{}.pt'.format(simnum)):
            try:
                # load memory dict from file
                memory_dict = torch.load(self.log + 'memory{}.pt'.format(simnum))
                # extend replay memory priority deque
                # print("REPLAY MEMORY:", memory_dict['replay_memory'])
                self.replay_memory.extend(memory_dict['replay_memory'])
                self.priority.extend(memory_dict['priority'])
                print('Memory loaded from ', self.log + 'memory{}.pt'.format(simnum))
                # clear that memory/priority from loaded dict --> prevent duplicates if loaded again?
                memory_dict['replay_memory'].clear()
                memory_dict['priority'].clear()
                torch.save(memory_dict, self.log + 'memory{}.pt'.format(simnum))
            except:
                time.sleep(10)
                memory_dict = torch.load(self.log + 'memory{}.pt'.format(simnum))
                self.replay_memory.extend(memory_dict['replay_memory'])
                self.priority.extend(memory_dict['priority'])
                print('Memory loaded from ', self.log + 'memory{}.pt'.format(simnum))
                memory_dict['replay_memory'].clear()
                memory_dict['priority'].clear()
                torch.save(memory_dict, self.log + 'memory{}.pt'.format(simnum))
        else:
            print("=> Learner: no memory found at ", self.log + 'memory{}.pt'.format(simnum))

    # Sampling experiencess based on importance
    def sample(self):
        # higher alpha --> more priorization
        priority = (np.array(self.priority) + self.e) ** self.alpha

        # beta slowly gets increased to 1 --> more uniform sampling
        weight = (len(priority) * priority) ** -self.beta
        # weight = map(lambda x: x ** -self.beta, (len(priority) * priority))

        # Normalize weights
        weight /= weight.max()
        self.weight = torch.tensor(weight, dtype=torch.float)
        priority = torch.tensor(priority, dtype=torch.float)
        return torch.utils.data.sampler.WeightedRandomSampler(priority, self.batch_size, replacement=True)

    def create_feature_mask(self, current_epoch):
        # Calculate num active channels using current epoch
        steps = (current_epoch // self.reduction_step)
        current_channels = max(self.min_channels, self.start_channels - steps)

        # Create mask with the used channels set to 1 and rest to 0
        mask = torch.ones((1, self.start_channels, 1, 1))
        if current_channels < self.start_channels:
            mask[:, current_channels:, :, :] = 0
        return mask.to(self.device)


    # train model
    def main(self):
        # save current model state and setting epoch
        train_epoch = self.start_epoch
        self.save_model(train_epoch)
        is_memory = False

        # check if enough replay memory. If not, try to load memory from files for each actor in system
        while len(self.replay_memory) < self.batch_size * 100:
            print("Memory not enough, current replay memory is: ", len(self.replay_memory))
            for i in range(self.actor_num):
                # print("SEARCHING FOR MEM:", self.log + '/memory{}.pt'.format(i))
                is_memory = os.path.isfile(self.log + '/memory{}.pt'.format(i))
                # print("IS_MEMORY:", i, is_memory)
                if is_memory:
                    self.load_memory(i)
                time.sleep(1)


        # train model        
        while True:
            self.optimizer.zero_grad()

            # set main dqn to train, target dqn to evaluate (ie. more stable)
            self.mainDQN.train()
            self.targetDQN.eval()

            # set of environment states sampled from replay memory --> fed into DQN
            x_stack = torch.zeros(0, self.history_size, self.height, self.width).to(self.device)
            # Q values corresponding to these environment states --> target
            y_stack = torch.zeros(0, self.action_size).to(self.device)

            w = []
            self.beta = min(1, self.beta_init + train_epoch * self.beta_increment)
            sample_idx = self.sample() # samples batch from replay memory

            # get feature mask
            feature_mask = self.create_feature_mask(train_epoch)

            # calculates temporal difference for each of the sampled experiences
            for idx in sample_idx:
                # get experience from replay memory
                history, action, reward, next_history, end = self.replay_memory[idx]
                # contains the 4 most recent frames
                history = history.to(self.device)
                next_history = next_history.to(self.device)

                # apply feature mask
                masked_history = history * feature_mask
                masked_next_history = next_history * feature_mask

                # assign Q value reward to the current state
                Q = self.mainDQN(masked_history)

                # if experience is the end of an episode, q value for action is set to received reward
                if end:
                    tderror = reward - Q[0, action]
                    Q[0, action] = reward

                # else, calculate TD error using the current q values from main DQN and target Q value from target DQN
                else:
                    # assign Q value reward by looking at Q value of the next state
                    qval = self.mainDQN(masked_next_history)
                    tderror = reward + self.dis * self.targetDQN(masked_next_history)[0, torch.argmax(qval, 1)] - Q[0, action]
                    Q[0, action] = reward + self.dis * self.targetDQN(masked_next_history)[0, torch.argmax(qval, 1)]
                
                # Add new state data
                x_stack = torch.cat([x_stack, masked_history.data], 0)
                y_stack = torch.cat([y_stack, Q.data], 0)
                w.append(self.weight[idx]) # weights for each experience?
                self.priority[idx] = tderror.abs().item() # larger TD --> sample more frequently
            

            pred = self.mainDQN(x_stack)    # predicted Q-values to be compared 
            w = torch.tensor(w, dtype=torch.float, device=self.device)  # weight for each experience

            # compute Smooth L1 loss (Huber loss) 
            loss = torch.dot(F.smooth_l1_loss(pred, y_stack.detach(), reduce=False).sum(1), w.detach())
            loss.backward()
            self.optimizer.step()
            loss /= self.batch_size # normalize to make it more comparable across batches of different size
            self.writer.add_scalar('loss', loss.item(), train_epoch)
            wandb.log({"loss": loss.item()}, step=train_epoch)

            # log the nstep
            n_step = max(self.min_channels, self.start_channels - train_epoch // self.reduction_step)
            wandb.log({"n_step": n_step}, step=train_epoch)

            train_epoch += 1
            gc.collect()

            # Log to console and self.writer --> Tensorboard
            if train_epoch % self.log_interval == 0:
                print('Train Epoch: {} \tLoss: {}'.format(train_epoch, loss.item()))
                self.writer.add_scalar('replay size', len(self.replay_memory), train_epoch)
                wandb.log({"replay size": len(self.replay_memory)}, step=train_epoch)
                if (train_epoch // self.log_interval) % args.actor_num == 0:
                    self.save_model(train_epoch)
                self.load_memory((train_epoch // self.log_interval) % args.actor_num) # loads mem for specific actor
            
            # Update target model --> provides stable Q-value targets for the main DQN to learn from
            if train_epoch % self.update_cycle == 0:
                self.update_target_model()


if __name__ == "__main__":
    learner = Learner()
    learner.main()
