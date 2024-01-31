import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os



class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions, name):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions


        self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)
        #q1_action_value = self.fc3(q1_action_value)
        #q1_action_value = F.relu(q1_action_value)
        
        q1 = self.q1(q1_action_value)

        return q1




class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions, name):
        super(ActorNetwork, self).__init__()

        self.filePath = 'agents/' + name
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        #prob = self.fc3(prob)
        #prob = F.relu(prob)

        mu = T.tanh(self.mu(prob))

        return mu

    def save_checkpoint(self, name):
        
        if not os.path.exists(self.filePath):
            os.mkdir(self.filePath) 

        T.save(self.state_dict(), self.filePath + '/' + name)


    def load_checkpoint(self, name):
        self.load_state_dict(T.load(self.filePath + '/' + name))


class agent():
    
    def __init__(self, conf):
        
        self.name = conf.name
        self.gamma = conf.gamma
        self.tau = conf.tau
        self.maxAction = 1
        self.minAction = -1
        input_dims = 24
        self.memory = ReplayBuffer(conf.maxReplaySize, input_dims, conf.numberActions)
        self.batch_size = conf.batchSize
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = conf.warmup
        self.n_actions = conf.numberActions
        self.update_actor_iter = conf.updateActorInterval
        self.maxVelocity = 5
        self.minVelocity = 1


        alpha = conf.alpha
        beta = conf.beta
        layer1Size = conf.layer1Size
        layer2Size = conf.layer2Size
        layer3Size = 300
        n_actions = conf.numberActions

        self.actor = ActorNetwork(alpha, input_dims, layer1Size, layer2Size, layer3Size, n_actions=n_actions, name=self.name)
        self.critic_1 = CriticNetwork(beta, input_dims, layer1Size, layer2Size, layer3Size, n_actions=n_actions, name=self.name)
        self.critic_2 = CriticNetwork(beta, input_dims, layer1Size, layer2Size, layer3Size, n_actions=n_actions, name=self.name)
        self.target_actor = ActorNetwork(alpha, input_dims, layer1Size, layer2Size, layer3Size, n_actions=n_actions, name=self.name)
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1Size, layer2Size, layer3Size, n_actions=n_actions, name=self.name)
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1Size, layer2Size, layer3Size, n_actions=n_actions, name=self.name)

        self.noise = conf.noise

        self.update_network_parameters(tau=1)



    
    def choose_action(self, observation, training):
        

        if (self.time_step<self.warmup) and training==True:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)))
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        
        if training==True:
            mu = mu.to(self.actor.device) + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)

        mu = T.clamp(mu, -1, 1)
        self.time_step += 1

        return mu.cpu().detach().numpy()


    
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return 

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        
        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, -1, 1)
        
        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + (1-tau)*target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

    def save_agent(self, name, run):
        self.actor.save_checkpoint(name + '_n_' + str(run))

    def load_weights(self, name, run):
        self.actor.load_checkpoint(name + '_n_' + str(run))

