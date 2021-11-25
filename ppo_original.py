import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import time
from utils import init_weights

CUDA_LAUNCH_BLOCKING=1



class PPOMemory:
    
    '''
    Constructor to create replay buffer. Returns batched numpy arrays for experience replay.
    
    Inputs:
    batch_size (int): mini batch size
    '''
    
    def __init__(self, batch_size):
        self.states = []
        self.probs = [] #log probs
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        

    def generate_batches(self):
        
        n_states = len(self.states)*2      
       
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        ##### needs work ~~~~~~
        return self.stack_tensor(np.array(self.states)),\
                self.stack_tensor(np.array(self.actions)),\
                self.stack_tensor(np.array(self.probs)),\
                self.stack_tensor(np.array(self.vals)),\
                self.stack_tensor(np.array(self.rewards)),\
                self.stack_tensor(np.array(self.dones)),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(np.squeeze(state))
        self.actions.append(np.squeeze(action))
        self.probs.append(np.squeeze(probs))
        self.vals.append(np.squeeze(vals))
        self.rewards.append(np.squeeze(reward))
        self.dones.append(np.squeeze(done))

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        
    def stack_tensor(self, some_list):
        return np.concatenate(some_list[:len(self.states)], axis=0)

    
    
class ActorNetwork(nn.Module):

    '''
    Constructor to create actor network to calculate action distribution function with two hidden layers. 
    Outputs a normal distribution for sampling.
    
    Inputs:
    n_actions (int): size of action space
    input_dims (int): size of inputs for neural networks
    alpha (float): Optimizer learning rate
    fc1_dims / fc2_dims (int): Hidden layer dimensions
    chkpt_dir (string): location to save network weights via checkpoint file
    
    Outputs:
    value (tensor): value function.
    '''
    
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=128, fc2_dims=64, chkpt_dir='solved_chkpt'):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Tanh()
        )

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.actor.to(self.device)
        
        #self.optimizer = optim.Adam(self.parameters(), lr=alpha)
  

        std_init = 0
        self.std = nn.Parameter(T.ones(n_actions) * std_init)
        
    def forward(self, state):
        mu = self.actor(state)

        std_scale = 1.0
        scaling = std_scale * F.softplus(self.std)
        dist = T.distributions.Normal(mu, scaling.to(self.device))
        
        return dist

    def save_checkpoint(self):
        T.save(self.actor.state_dict(), 'actor_torch_ppo.pth')

    def load_checkpoint(self):
        self.actor.load_state_dict(T.load('actor_torch_ppo.pth'))

        
class CriticNetwork(nn.Module):

    '''
    Constructor to create critic network to calculate value function with two hidden layers. Outputs single value.
    
    Inputs:
    input_dims (int): size of inputs for neural networks
    alpha (float): Optimizer learning rate
    fc1_dims / fc2_dims (int): Hidden layer dimensions
    chkpt_dir (string): location to save network weights via checkpoint file
    
    Outputs:
    value (tensor): value function.
    '''
    
    def __init__(self, input_dims, alpha, fc1_dims=128, fc2_dims=64,
            chkpt_dir='solved_chkpt'):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.critic.to(self.device)
        
        #self.optimizer = optim.Adam(self.parameters(), lr=alpha)
   
    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.critic.state_dict(), 'critic_torch_ppo.pth')

    def load_checkpoint(self):
        self.critic.load_state_dict(T.load('critic_torch_ppo.pth'))

        

   
    
        
class Agent:
    
    '''
    Constructor to create PPO actor critic model.
    
    Inputs:
    n_actions (int): size of action space
    input_dims (int): size of inputs for neural networks
    gamma (float): discount factor
    alpha (float): Optimizer learning rate
    gae_lambda (float): GAE smoothing parameter
    policy_clip (float) PPO policy loss function clipping parameter
    batch_size (int): mini batch size
    n_epochs (int): number of epochs, or training cycles per update
    value_loss_weight (float): Weighting parameter for value loss in total loss function
    '''
    
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0005, gae_lambda=0.9,
            policy_clip=0.2, batch_size=256, n_epochs=16, value_loss_weight=1.0, n_agents=1,
                gradient_clip = 0):

        
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.value_loss_weight = value_loss_weight
        
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
        
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
        
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr = alpha)
               
        self.gradient_clip = gradient_clip
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.n_agents = n_agents

        
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        T.save(self.actor.state_dict(), 'actor_torch_ppo.pth')
        T.save(self.critic.state_dict(), 'critic_torch_ppo.pth')

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.actor.eval()
        self.critic.eval()

    def choose_action(self, states):
        '''
        Determine action from current policy (actor network) and calculate associated logarithm of the 
        probability and calculate associated value function value.
        
        Inputs:
        states (np array): Environmental obervations 
        
        '''
        
        states = T.from_numpy(np.stack(states)).float().to(self.device)
        dist = self.actor(states)
        value = self.critic(states)

        action = dist.sample()

        probs = dist.log_prob(action)
        
               
        return action.cpu().detach().numpy(), probs.cpu().detach().numpy(), value.cpu().detach().numpy()


    def get_actions(self, states):
        
        actions = [self.choose_action(states) for agent in agents]
        
        return actions

    
    def learn(self, n_steps):

        '''
        Function to generate a batch of experiences, calculate advantage and loss functions and then perform back propogation to 
        update the actor and critic models.
                
        '''
        
        for _ in range(self.n_epochs):
            
            adv_time = time.time()
            
            # Generate batches
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()


            # Initialize advantage function
            advantage = np.zeros(len(reward_arr))
            
            # For every timestep within batch
            for t in range(len(reward_arr)-1):
                
                # discount starts at one for first delta_t term
                discount = 1
                
                # Initialize timestep advantage value
            
            a_t = 0
                
            # Calculate advantage using GAE    
            for i in reversed(range(len(reward_arr) - 1)):

                td = reward_arr[i] + self.gamma * vals_arr[i + 1]*(1 - dones_arr[i]) - vals_arr[i]
                a_t = self.gamma * self.gae_lambda * (1 - dones_arr[i]) * a_t + td

            
                # Add advantage value to overall
                advantage[i] = a_t

            #Normalize advantages
            #advantage = (advantage - advantage.mean()) / advantage.std() + 1.0e-10
            
            advantage = T.tensor(advantage).to(self.device)
            
            values = T.tensor(vals_arr).to(self.device)

            
            # Loop over each batch
            for batch in batches:
                  
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.device)

                
                old_probs = T.tensor(old_prob_arr[batch]).to(self.device)

                actions = T.tensor(action_arr[batch]).to(self.device)

                dist = self.actor(states)
                
                #entropy = T.mean(dist.entropy())
                
                critic_value = self.critic(states)
                
                
                new_probs = dist.log_prob(actions) # new log_probs

                prob_ratio = (new_probs - old_probs).sum(-1).exp()
                
                weighted_probs = advantage[batch] * prob_ratio
                
                
                
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                
                # Clipped actor loss
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                #print('advantage shape: {}'.format(advantage[batch].shape))
                returns = advantage[batch] + values[batch]

                # Calculate value loss
                critic_loss = (returns-np.squeeze(critic_value))**2
                critic_loss = critic_loss.mean()

                # Calculate total loss
                total_loss = actor_loss + self.value_loss_weight*critic_loss# - self.entropy_coeff*entropy

                self.optimizer.zero_grad()
                
                # Update actor and critic networks
                total_loss.backward()
                               
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
                
                self.optimizer.step()
                #self.actor.optimizer.step()
                #self.critic.optimizer.step()
                #self.entropy_coeff *= self.entropy_decay
        
        self.memory.clear_memory()
        
       

