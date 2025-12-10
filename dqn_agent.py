"""
Deep Q-Network (DQN) Agent for Agent Coordination Learning

Implements value-based reinforcement learning to learn optimal
agent coordination strategies in multi-agent content creation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import logging

logger = logging.getLogger(__name__)

# Experience replay tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class QNetwork(nn.Module):
    """Deep Q-Network for value function approximation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 128, 64]):
        super(QNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for learning agent coordination"""
    
    def __init__(self, 
                state_dim: int,
                action_dim: int,
                learning_rate: float = 0.001,
                gamma: float = 0.95,
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.01,
                epsilon_decay: float = 0.995,
                buffer_size: int = 10000,
                batch_size: int = 64,
                target_update_freq: int = 10):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Q-Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.training_step = 0
        self.losses = []
        
        logger.info(f"DQN Agent initialized - State dim: {state_dim}, "
                   f"Action dim: {action_dim}, Device: {self.device}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randrange(self.action_dim)
        
        # Exploit: best action from Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self) -> float:
        """Train the DQN agent"""
        
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            )
            target_q_values = rewards.unsqueeze(1) + \
                             (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_network(state_tensor).cpu().numpy()[0]
    
    def save(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
        logger.info(f"DQN model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        logger.info(f"DQN model loaded from {filepath}")
    
    def get_state(self) -> dict:
        """Get current agent state"""
        return {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0
        }


class StateEncoder:
    """Encodes agent system state into vector representation"""
    
    def __init__(self, num_agents: int, num_task_types: int):
        self.num_agents = num_agents
        self.num_task_types = num_task_types
        self.state_dim = self._calculate_state_dim()
    
    def _calculate_state_dim(self) -> int:
        """Calculate total state dimension"""
        # Task features: type (one-hot) + complexity + requirements
        task_dim = self.num_task_types + 5
        
        # Agent features: performance history + availability
        agent_dim = self.num_agents * 3
        
        # Context features: time, resources, history
        context_dim = 10
        
        return task_dim + agent_dim + context_dim
    
    def encode(self, task: dict, agent_states: dict, context: dict) -> np.ndarray:
        """Encode current state"""
        
        # Task encoding
        task_type_onehot = self._encode_task_type(task['type'])
        task_complexity = self._estimate_complexity(task)
        task_features = np.concatenate([
            task_type_onehot,
            [task_complexity],
            self._encode_requirements(task.get('requirements', {}))
        ])
        
        # Agent encoding
        agent_features = self._encode_agents(agent_states)
        
        # Context encoding
        context_features = self._encode_context(context)
        
        # Concatenate all features
        state = np.concatenate([task_features, agent_features, context_features])
        
        return state.astype(np.float32)
    
    def _encode_task_type(self, task_type: str) -> np.ndarray:
        """One-hot encode task type"""
        task_types = ['blog_post', 'technical_article', 'marketing_copy', 'research_summary', 'tutorial']
        encoding = np.zeros(self.num_task_types)
        if task_type in task_types:
            encoding[task_types.index(task_type)] = 1.0
        return encoding
    
    def _estimate_complexity(self, task: dict) -> float:
        """Estimate task complexity (0-1)"""
        requirements = task.get('requirements', {})
        length = requirements.get('length', 500)
        normalized_length = min(length / 2000, 1.0)
        return normalized_length
    
    def _encode_requirements(self, requirements: dict) -> np.ndarray:
        """Encode task requirements"""
        # Length, tone complexity, audience specificity, constraints
        features = np.zeros(4)
        
        # Normalized length
        features[0] = min(requirements.get('length', 500) / 2000, 1.0)
        
        # Tone complexity
        tone_complexity = {
            'informative': 0.3, 'technical': 0.8, 'persuasive': 0.6,
            'academic': 0.9, 'educational': 0.5, 'casual': 0.2
        }
        features[1] = tone_complexity.get(requirements.get('tone', 'informative'), 0.5)
        
        # Audience specificity
        audience_specificity = {
            'general': 0.3, 'developers': 0.7, 'business': 0.6,
            'researchers': 0.9, 'students': 0.5
        }
        features[2] = audience_specificity.get(
            requirements.get('target_audience', 'general'), 0.5
        )
        
        # Additional constraints
        features[3] = len(requirements) / 10.0
        
        return features
    
    def _encode_agents(self, agent_states: dict) -> np.ndarray:
        """Encode agent states"""
        features = []
        for agent_id in sorted(agent_states.keys())[:self.num_agents]:
            state = agent_states[agent_id]
            features.extend([
                state.get('success_rate', 0.5),
                state.get('avg_quality', 0.5),
                1.0 if state.get('available', True) else 0.0
            ])
        
        # Pad if necessary
        while len(features) < self.num_agents * 3:
            features.extend([0.5, 0.5, 1.0])
        
        return np.array(features[:self.num_agents * 3])
    
    def _encode_context(self, context: dict) -> np.ndarray:
        """Encode contextual information"""
        features = np.zeros(10)
        
        # Episode progress
        features[0] = context.get('episode_progress', 0.0)
        
        # Recent performance
        features[1] = context.get('recent_avg_reward', 0.5)
        features[2] = context.get('recent_avg_quality', 0.5)
        
        # Resource availability
        features[3] = context.get('resource_availability', 1.0)
        
        # Exploration rate
        features[4] = context.get('exploration_rate', 0.5)
        
        # Performance trend (improving/declining)
        features[5] = context.get('performance_trend', 0.0)
        
        # Agent coordination history
        features[6:10] = context.get('coordination_history', [0.5] * 4)
        
        return features