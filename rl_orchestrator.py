"""
RL-Enhanced Agent Orchestrator

Coordinates multiple specialized agents using reinforcement learning
to optimize task execution and learn effective coordination strategies.

"""

import time
import logging
from typing import Dict, List
import numpy as np

from dqn_agent import DQNAgent, StateEncoder
from thompson_sampler import ThompsonSampler, UCBAgent, HybridExploration
from agents import AgentTeam
from reward_function import RewardCalculator

logger = logging.getLogger(__name__)


class RLAgentOrchestrator:
    """Orchestrates agents using RL for optimal coordination"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize agent team
        self.agent_team = AgentTeam(config)
        num_agents = len(self.agent_team.agents)
        
        # State encoder
        self.state_encoder = StateEncoder(
            num_agents=num_agents,
            num_task_types=5
        )
        
        # DQN for learning coordination strategies
        self.num_coordination_patterns = config.num_coordination_patterns
        self.dqn_agent = DQNAgent(
            state_dim=self.state_encoder.state_dim,
            action_dim=self.num_coordination_patterns,
            learning_rate=config.dqn_learning_rate,
            gamma=config.dqn_gamma,
            epsilon_start=config.dqn_epsilon_start,
            epsilon_end=config.dqn_epsilon_end,
            epsilon_decay=config.dqn_epsilon_decay
        )
        
        # Thompson Sampling for agent selection exploration
        self.thompson_sampler = HybridExploration(
            num_agents=num_agents,
            thompson_weight=config.thompson_weight
        )
        
        # Reward calculator
        self.reward_calculator = RewardCalculator()
        
        # Agent performance tracking
        self.agent_states = {
            agent_id: {
                'success_rate': 0.5,
                'avg_quality': 0.5,
                'available': True,
                'total_tasks': 0,
                'successful_tasks': 0
            }
            for agent_id in range(num_agents)
        }
        
        # Episode context
        self.episode = 0
        self.recent_rewards = []
        self.recent_qualities = []
        
        # Pattern usage tracking (FIX #1)
        self.pattern_usage = {i: 0 for i in range(self.num_coordination_patterns)}
        self.pattern_rewards = {i: [] for i in range(self.num_coordination_patterns)}
        
        logger.info(f"RL Orchestrator initialized with {num_agents} agents")
    
    def execute_task(self, task: dict) -> dict:
        """
        Execute task with RL-guided agent coordination
        
        FIXED: Now includes forced exploration and pattern diversity tracking
        """
        start_time = time.time()
        
        # Encode current state
        context = self._build_context()
        state = self.state_encoder.encode(task, self.agent_states, context)
        
        # Select coordination pattern (FIX #2: Forced exploration)
        coordination_pattern = self._select_coordination_pattern(state)
        
        # Track pattern usage
        self.pattern_usage[coordination_pattern] += 1
        
        # Execute task with selected pattern
        result = self._execute_with_pattern(
            task, coordination_pattern, context
        )
        
        # Calculate reward (FIX #3: Include pattern diversity bonus)
        reward = self.reward_calculator.calculate_reward(
            task=task,
            result=result,
            execution_time=result['execution_time'],
            coordination_pattern=coordination_pattern,
            pattern_usage=self.pattern_usage,
            episode=self.episode
        )
        
        # Track pattern performance
        self.pattern_rewards[coordination_pattern].append(reward)
        
        # Get next state
        self._update_agent_states(result)
        next_context = self._build_context()
        next_state = self.state_encoder.encode(task, self.agent_states, next_context)
        
        # Store experience and train DQN
        done = False
        self.dqn_agent.store_experience(state, coordination_pattern, reward, next_state, done)
        loss = self.dqn_agent.train()
        
        # Update Thompson Sampler
        for agent_id, agent_reward in result.get('agent_rewards', {}).items():
            self.thompson_sampler.update(
                agent_id, 
                agent_reward,
                success=(agent_reward > 0.6)
            )
        
        # Track metrics
        self.recent_rewards.append(reward)
        self.recent_qualities.append(result['quality_score'])
        if len(self.recent_rewards) > 20:
            self.recent_rewards.pop(0)
            self.recent_qualities.pop(0)
        
        self.episode += 1
        
        execution_time = time.time() - start_time
        
        return {
            'reward': reward,
            'quality_score': result['quality_score'],
            'execution_time': execution_time,
            'agent_sequence': result['agent_sequence'],
            'coordination_pattern': coordination_pattern,
            'dqn_loss': loss,
            'content': result.get('content', ''),
            'metrics': result.get('metrics', {})
        }
    
    def _select_coordination_pattern(self, state: np.ndarray) -> int:
        """
        Select coordination pattern with forced exploration
        
        FIX: First 30 episodes force exploration of all patterns
        """
        # Phase 1: Forced exploration (episodes 0-29)
        if self.episode < 30:
            # Cycle through all patterns multiple times
            pattern = self.episode % self.num_coordination_patterns
            logger.debug(f"Episode {self.episode}: Forced pattern {pattern}")
            return pattern
        
        # Phase 2: Guided exploration (episodes 30-59)
        elif self.episode < 60:
            # 50% DQN, 50% random (to ensure more exploration)
            if np.random.random() < 0.5:
                pattern = self.dqn_agent.select_action(state, training=True)
            else:
                pattern = np.random.randint(0, self.num_coordination_patterns)
            logger.debug(f"Episode {self.episode}: Guided exploration pattern {pattern}")
            return pattern
        
        # Phase 3: Normal DQN with epsilon-greedy (episodes 60+)
        else:
            pattern = self.dqn_agent.select_action(state, training=True)
            logger.debug(f"Episode {self.episode}: DQN selected pattern {pattern} (ε={self.dqn_agent.epsilon:.3f})")
            return pattern
    
    def _execute_with_pattern(self, task: dict, pattern: int, context: dict) -> dict:
        """Execute task using specified coordination pattern"""
        start_time = time.time()
        
        if pattern == 0:
            result = self._sequential_execution(task)
        elif pattern == 1:
            result = self._parallel_execution(task)
        elif pattern == 2:
            result = self._hierarchical_execution(task)
        elif pattern == 3:
            result = self._collaborative_execution(task)
        else:  # pattern == 4
            result = self._adaptive_execution(task)
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def _sequential_execution(self, task: dict) -> dict:
        """Execute with agents in sequence"""
        logger.debug("Executing with Sequential pattern")
        
        agent_sequence = self.agent_team.get_task_sequence(task['type'])
        
        content = ""
        quality_scores = []
        agent_rewards = {}
        
        for i, agent in enumerate(agent_sequence):
            agent_id = self.agent_team.get_agent_id(agent)
            agent_result = agent.execute(task, previous_content=content)
            content = agent_result['content']
            quality = agent_result['quality']
            quality_scores.append(quality)
            agent_rewards[agent_id] = quality
        
        return {
            'content': content,
            'quality_score': np.mean(quality_scores),
            'agent_sequence': [self.agent_team.get_agent_id(a) for a in agent_sequence],
            'agent_rewards': agent_rewards,
            'metrics': {'pattern': 'sequential'}
        }
    
    def _parallel_execution(self, task: dict) -> dict:
        """Execute with agents working in parallel"""
        logger.debug("Executing with Parallel pattern")
        
        agents = self.agent_team.get_task_agents(task['type'])
        results = []
        agent_rewards = {}
        
        for agent in agents:
            agent_id = self.agent_team.get_agent_id(agent)
            agent_result = agent.execute(task)
            results.append(agent_result)
            agent_rewards[agent_id] = agent_result['quality']
        
        best_result = max(results, key=lambda x: x['quality'])
        
        return {
            'content': best_result['content'],
            'quality_score': best_result['quality'],
            'agent_sequence': [self.agent_team.get_agent_id(a) for a in agents],
            'agent_rewards': agent_rewards,
            'metrics': {'pattern': 'parallel'}
        }
    
    def _hierarchical_execution(self, task: dict) -> dict:
        """Execute with lead agent coordinating others"""
        logger.debug("Executing with Hierarchical pattern")
        
        agents = self.agent_team.agents
        lead_id, _ = self.thompson_sampler.select_agent()
        lead_agent = agents[lead_id]
        
        lead_result = lead_agent.execute(task)
        content = lead_result['content']
        agent_rewards = {lead_id: lead_result['quality']}
        quality_scores = [lead_result['quality']]
        
        for i, agent in enumerate(agents):
            if i != lead_id:
                refinement = agent.execute(task, previous_content=content)
                content = refinement['content']
                quality_scores.append(refinement['quality'])
                agent_rewards[i] = refinement['quality']
        
        return {
            'content': content,
            'quality_score': np.mean(quality_scores),
            'agent_sequence': [lead_id] + [i for i in range(len(agents)) if i != lead_id],
            'agent_rewards': agent_rewards,
            'metrics': {'pattern': 'hierarchical', 'lead_agent': lead_id}
        }
    
    def _collaborative_execution(self, task: dict) -> dict:
        """Execute with agents collaborating iteratively"""
        logger.debug("Executing with Collaborative pattern")
        
        agents = self.agent_team.get_task_agents(task['type'])
        content = ""
        iterations = 2
        agent_rewards = {}
        quality_scores = []
        
        for iteration in range(iterations):
            for agent in agents:
                agent_id = self.agent_team.get_agent_id(agent)
                result = agent.execute(task, previous_content=content)
                content = result['content']
                quality_scores.append(result['quality'])
                agent_rewards[agent_id] = result['quality']
        
        return {
            'content': content,
            'quality_score': np.mean(quality_scores[-len(agents):]),
            'agent_sequence': [self.agent_team.get_agent_id(a) for a in agents] * iterations,
            'agent_rewards': agent_rewards,
            'metrics': {'pattern': 'collaborative', 'iterations': iterations}
        }
    
    def _adaptive_execution(self, task: dict) -> dict:
        """Execute with adaptive agent selection via Thompson Sampling"""
        logger.debug("Executing with Adaptive (Thompson Sampling) pattern")
        
        agents = self.agent_team.agents
        content = ""
        num_steps = 3
        
        agent_sequence = []
        agent_rewards = {}
        quality_scores = []
        
        for step in range(num_steps):
            agent_id, strategy = self.thompson_sampler.select_agent()
            agent = agents[agent_id]
            
            result = agent.execute(task, previous_content=content)
            content = result['content']
            quality = result['quality']
            
            agent_sequence.append(agent_id)
            quality_scores.append(quality)
            agent_rewards[agent_id] = quality
        
        return {
            'content': content,
            'quality_score': np.mean(quality_scores),
            'agent_sequence': agent_sequence,
            'agent_rewards': agent_rewards,
            'metrics': {'pattern': 'adaptive', 'steps': num_steps}
        }
    
    def _build_context(self) -> dict:
        """Build context for state encoding"""
        return {
            'episode_progress': min(self.episode / 200, 1.0),  # Changed to 200
            'recent_avg_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0.5,
            'recent_avg_quality': np.mean(self.recent_qualities) if self.recent_qualities else 0.5,
            'resource_availability': 1.0,
            'exploration_rate': self.dqn_agent.epsilon,
            'performance_trend': self._calculate_trend(),
            'coordination_history': self._get_coordination_history()
        }
    
    def _calculate_trend(self) -> float:
        """Calculate performance trend (-1 to 1)"""
        if len(self.recent_rewards) < 10:
            return 0.0
        
        recent = self.recent_rewards[-5:]
        earlier = self.recent_rewards[-10:-5]
        
        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier)
        
        return np.clip((recent_avg - earlier_avg) / 0.5, -1, 1)
    
    def _get_coordination_history(self) -> list:
        """Get recent coordination pattern usage"""
        total = sum(self.pattern_usage.values())
        if total == 0:
            return [0.2] * 4
        
        # Return normalized usage for first 4 patterns
        history = [self.pattern_usage[i] / max(total, 1) for i in range(4)]
        return history
    
    def _update_agent_states(self, result: dict):
        """Update agent performance tracking"""
        for agent_id, reward in result.get('agent_rewards', {}).items():
            state = self.agent_states[agent_id]
            state['total_tasks'] += 1
            
            if reward > 0.6:
                state['successful_tasks'] += 1
            
            state['success_rate'] = state['successful_tasks'] / state['total_tasks']
            
            alpha = 0.1
            state['avg_quality'] = alpha * reward + (1 - alpha) * state['avg_quality']
    
    def save_models(self, filepath_prefix: str):
        """Save RL models"""
        self.dqn_agent.save(f"{filepath_prefix}_dqn.pth")
        logger.info(f"Models saved to {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str):
        """Load RL models"""
        self.dqn_agent.load(f"{filepath_prefix}_dqn.pth")
        logger.info(f"Models loaded from {filepath_prefix}")
    
    def get_pattern_statistics(self) -> dict:
        """Get pattern usage statistics"""
        stats = {}
        for pattern_id in range(self.num_coordination_patterns):
            pattern_names = ['Sequential', 'Parallel', 'Hierarchical', 'Collaborative', 'Adaptive']
            stats[pattern_names[pattern_id]] = {
                'usage_count': self.pattern_usage[pattern_id],
                'avg_reward': np.mean(self.pattern_rewards[pattern_id]) if self.pattern_rewards[pattern_id] else 0.0,
                'usage_percentage': (self.pattern_usage[pattern_id] / max(sum(self.pattern_usage.values()), 1)) * 100
            }
        return stats