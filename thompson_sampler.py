"""
Thompson Sampling and UCB for Exploration in Agent Selection

Implements contextual bandits with Thompson Sampling and 
Upper Confidence Bound (UCB) for intelligent exploration.
"""

import numpy as np
from scipy import stats
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ThompsonSampler:
    """
    Thompson Sampling for agent selection with contextual bandits
    """
    
    def __init__(self, num_agents: int, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.num_agents = num_agents
        
        # Beta distribution parameters for each agent
        self.alpha = np.ones(num_agents) * alpha_prior
        self.beta = np.ones(num_agents) * beta_prior
        
        # Track selections and rewards
        self.selections = np.zeros(num_agents)
        self.successes = np.zeros(num_agents)
        self.rewards_history = [[] for _ in range(num_agents)]
        
        logger.info(f"Thompson Sampler initialized for {num_agents} agents")
    
    def sample_agent(self) -> int:
        """
        Sample an agent using Thompson Sampling
        
        Returns:
            Agent index to select
        """
        # Sample from each agent's beta distribution
        samples = np.random.beta(self.alpha, self.beta)
        
        # Select agent with highest sample
        selected_agent = np.argmax(samples)
        
        return int(selected_agent)
    
    def update(self, agent_id: int, reward: float, success: bool = None):
        """
        Update agent statistics after observing reward
        
        Args:
            agent_id: Agent that was selected
            reward: Observed reward (0-1)
            success: Whether task was successful (optional, inferred from reward)
        """
        self.selections[agent_id] += 1
        self.rewards_history[agent_id].append(reward)
        
        # Determine success (reward > threshold or explicit)
        if success is None:
            success = reward > 0.6
        
        # Update beta distribution parameters
        if success:
            self.alpha[agent_id] += 1
            self.successes[agent_id] += 1
        else:
            self.beta[agent_id] += 1
        
        logger.debug(f"Agent {agent_id} updated: α={self.alpha[agent_id]:.2f}, "
                    f"β={self.beta[agent_id]:.2f}")
    
    def get_agent_stats(self, agent_id: int) -> Dict:
        """Get statistics for an agent"""
        if self.selections[agent_id] == 0:
            return {
                'mean_reward': 0.5,
                'success_rate': 0.5,
                'uncertainty': 1.0,
                'selections': 0
            }
        
        mean_reward = np.mean(self.rewards_history[agent_id])
        success_rate = self.successes[agent_id] / self.selections[agent_id]
        
        # Calculate uncertainty (variance of beta distribution)
        a, b = self.alpha[agent_id], self.beta[agent_id]
        variance = (a * b) / ((a + b) ** 2 * (a + b + 1))
        
        return {
            'mean_reward': float(mean_reward),
            'success_rate': float(success_rate),
            'uncertainty': float(variance),
            'selections': int(self.selections[agent_id]),
            'alpha': float(a),
            'beta': float(b)
        }
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all agents"""
        return {
            f'agent_{i}': self.get_agent_stats(i) 
            for i in range(self.num_agents)
        }
    
    def get_state(self) -> dict:
        """Get current sampler state"""
        return {
            'alpha': self.alpha.tolist(),
            'beta': self.beta.tolist(),
            'selections': self.selections.tolist(),
            'successes': self.successes.tolist()
        }


class UCBAgent:
    """
    Upper Confidence Bound (UCB1) for agent selection
    """
    
    def __init__(self, num_agents: int, c: float = 2.0):
        self.num_agents = num_agents
        self.c = c  # Exploration parameter
        
        # Track selections and rewards
        self.selections = np.zeros(num_agents)
        self.total_reward = np.zeros(num_agents)
        self.rewards_history = [[] for _ in range(num_agents)]
        
        self.total_selections = 0
        
        logger.info(f"UCB Agent initialized for {num_agents} agents with c={c}")
    
    def select_agent(self) -> int:
        """
        Select agent using UCB1 algorithm
        
        Returns:
            Agent index to select
        """
        # Initially select each agent once
        if self.total_selections < self.num_agents:
            return self.total_selections
        
        # Calculate UCB values
        ucb_values = np.zeros(self.num_agents)
        
        for i in range(self.num_agents):
            if self.selections[i] == 0:
                ucb_values[i] = float('inf')
            else:
                # Mean reward
                mean_reward = self.total_reward[i] / self.selections[i]
                
                # Confidence bound
                confidence = self.c * np.sqrt(
                    np.log(self.total_selections) / self.selections[i]
                )
                
                ucb_values[i] = mean_reward + confidence
        
        # Select agent with highest UCB value
        selected_agent = np.argmax(ucb_values)
        
        return int(selected_agent)
    
    def update(self, agent_id: int, reward: float):
        """Update agent statistics"""
        self.selections[agent_id] += 1
        self.total_reward[agent_id] += reward
        self.rewards_history[agent_id].append(reward)
        self.total_selections += 1
        
        logger.debug(f"Agent {agent_id} updated: reward={reward:.3f}, "
                    f"total_selections={self.selections[agent_id]}")
    
    def get_agent_stats(self, agent_id: int) -> Dict:
        """Get statistics for an agent"""
        if self.selections[agent_id] == 0:
            return {
                'mean_reward': 0.0,
                'ucb_value': float('inf'),
                'selections': 0
            }
        
        mean_reward = self.total_reward[agent_id] / self.selections[agent_id]
        
        # Calculate confidence bound
        if self.total_selections > 0:
            confidence = self.c * np.sqrt(
                np.log(self.total_selections) / self.selections[agent_id]
            )
            ucb_value = mean_reward + confidence
        else:
            ucb_value = float('inf')
        
        return {
            'mean_reward': float(mean_reward),
            'ucb_value': float(ucb_value),
            'selections': int(self.selections[agent_id])
        }
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all agents"""
        return {
            f'agent_{i}': self.get_agent_stats(i) 
            for i in range(self.num_agents)
        }


class ContextualBandit:
    """
    Contextual Bandit combining context with Thompson Sampling
    """
    
    def __init__(self, num_agents: int, context_dim: int, learning_rate: float = 0.01):
        self.num_agents = num_agents
        self.context_dim = context_dim
        self.learning_rate = learning_rate
        
        # Linear models for each agent: θ_i for context features
        self.theta = np.zeros((num_agents, context_dim))
        
        # Covariance matrices for uncertainty estimation
        self.A = [np.identity(context_dim) for _ in range(num_agents)]
        self.b = [np.zeros(context_dim) for _ in range(num_agents)]
        
        self.selections = np.zeros(num_agents)
        
        logger.info(f"Contextual Bandit initialized: {num_agents} agents, "
                   f"{context_dim} context dimensions")
    
    def select_agent(self, context: np.ndarray, use_ucb: bool = True) -> int:
        """
        Select agent based on context
        
        Args:
            context: Context vector
            use_ucb: Whether to use UCB for exploration
            
        Returns:
            Selected agent index
        """
        scores = np.zeros(self.num_agents)
        
        for i in range(self.num_agents):
            # Predicted reward
            predicted_reward = np.dot(self.theta[i], context)
            
            if use_ucb and self.selections[i] > 0:
                # Add exploration bonus
                A_inv = np.linalg.inv(self.A[i])
                uncertainty = np.sqrt(np.dot(context.T, np.dot(A_inv, context)))
                scores[i] = predicted_reward + uncertainty
            else:
                scores[i] = predicted_reward
        
        selected_agent = np.argmax(scores)
        self.selections[selected_agent] += 1
        
        return int(selected_agent)
    
    def update(self, agent_id: int, context: np.ndarray, reward: float):
        """
        Update contextual model for agent
        
        Args:
            agent_id: Selected agent
            context: Context vector
            reward: Observed reward
        """
        # Update A and b matrices (linear regression with L2 regularization)
        self.A[agent_id] += np.outer(context, context)
        self.b[agent_id] += reward * context
        
        # Update theta (ridge regression solution)
        try:
            A_inv = np.linalg.inv(self.A[agent_id])
            self.theta[agent_id] = np.dot(A_inv, self.b[agent_id])
        except np.linalg.LinAlgError:
            logger.warning(f"Singular matrix for agent {agent_id}, using regularization")
            A_reg = self.A[agent_id] + 0.1 * np.identity(self.context_dim)
            A_inv = np.linalg.inv(A_reg)
            self.theta[agent_id] = np.dot(A_inv, self.b[agent_id])
        
        logger.debug(f"Contextual model updated for agent {agent_id}")
    
    def get_agent_predictions(self, context: np.ndarray) -> np.ndarray:
        """Get predicted rewards for all agents given context"""
        predictions = np.array([
            np.dot(self.theta[i], context) 
            for i in range(self.num_agents)
        ])
        return predictions


class HybridExploration:
    """
    Hybrid exploration strategy combining Thompson Sampling and UCB
    """
    
    def __init__(self, num_agents: int, thompson_weight: float = 0.5):
        self.thompson = ThompsonSampler(num_agents)
        self.ucb = UCBAgent(num_agents)
        self.thompson_weight = thompson_weight
        
        self.num_agents = num_agents
        self.strategy_uses = {'thompson': 0, 'ucb': 0}
        
        logger.info(f"Hybrid Exploration initialized with weight={thompson_weight}")
    
    def select_agent(self, use_thompson: bool = None) -> Tuple[int, str]:
        """
        Select agent using hybrid strategy
        
        Args:
            use_thompson: Force strategy (None for automatic)
            
        Returns:
            (agent_id, strategy_used)
        """
        if use_thompson is None:
            use_thompson = np.random.random() < self.thompson_weight
        
        if use_thompson:
            agent = self.thompson.sample_agent()
            strategy = 'thompson'
            self.strategy_uses['thompson'] += 1
        else:
            agent = self.ucb.select_agent()
            strategy = 'ucb'
            self.strategy_uses['ucb'] += 1
        
        return agent, strategy
    
    def update(self, agent_id: int, reward: float, success: bool = None):
        """Update both samplers"""
        self.thompson.update(agent_id, reward, success)
        self.ucb.update(agent_id, reward)
    
    def get_all_stats(self) -> Dict:
        """Get combined statistics"""
        return {
            'thompson': self.thompson.get_all_stats(),
            'ucb': self.ucb.get_all_stats(),
            'strategy_uses': self.strategy_uses
        }
    
    def get_state(self) -> dict:
        """Get current state"""
        return {
            'thompson_state': self.thompson.get_state(),
            'strategy_uses': self.strategy_uses
        }