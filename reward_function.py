"""
Reward Function Engineering for Agent Coordination Learning

Defines reward structure to encourage optimal agent coordination,
quality content generation, and efficient execution.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class RewardCalculator:
    """Calculates rewards for agent coordination and task execution"""
    
    def __init__(self,
            quality_weight: float = 0.4,  # Reduced from 0.5
            efficiency_weight: float = 0.2,
            coordination_weight: float = 0.2,
            diversity_weight: float = 0.2):  # Increased from 0.1
        
        self.quality_weight = quality_weight
        self.efficiency_weight = efficiency_weight
        self.coordination_weight = coordination_weight
        self.diversity_weight = diversity_weight
        
        # Normalize weights
        total = sum([quality_weight, efficiency_weight, 
                    coordination_weight, diversity_weight])
        self.quality_weight /= total
        self.efficiency_weight /= total
        self.coordination_weight /= total
        self.diversity_weight /= total
        
        logger.info(f"Reward Calculator initialized with weights: "
                f"quality={self.quality_weight:.2f}, "
                f"efficiency={self.efficiency_weight:.2f}, "
                f"coordination={self.coordination_weight:.2f}, "
                f"diversity={self.diversity_weight:.2f}")
    
    def calculate_reward(self,
                        task: dict,
                        result: dict,
                        execution_time: float,
                        coordination_pattern: int,
                        pattern_usage: dict = None,
                        episode: int = 0) -> float:
        """
        Calculate comprehensive reward for task execution
        
        FIXED: Now includes pattern diversity bonus
        """
        
        # Component rewards
        quality_reward = self._calculate_quality_reward(result)
        efficiency_reward = self._calculate_efficiency_reward(execution_time)
        coordination_reward = self._calculate_coordination_reward(result)
        diversity_reward = self._calculate_diversity_reward(
            coordination_pattern, pattern_usage, episode
        )
        
        # Weighted combination
        total_reward = (
            self.quality_weight * quality_reward +
            self.efficiency_weight * efficiency_reward +
            self.coordination_weight * coordination_reward +
            self.diversity_weight * diversity_reward
        )
        
        logger.debug(f"Reward breakdown - Quality: {quality_reward:.3f}, "
                    f"Efficiency: {efficiency_reward:.3f}, "
                    f"Coordination: {coordination_reward:.3f}, "
                    f"Diversity: {diversity_reward:.3f}, "
                    f"Total: {total_reward:.3f}")
        
        return total_reward
    
    def _calculate_quality_reward(self, result: dict) -> float:
        """Calculate reward based on content quality"""
        quality_score = result.get('quality_score', 0.5)
        quality_reward = quality_score ** 1.5
        return quality_reward
    
    def _calculate_efficiency_reward(self, execution_time: float) -> float:
        """Calculate reward based on execution efficiency"""
        optimal_time = 5.0
        acceptable_time = 15.0
        
        if execution_time <= optimal_time:
            efficiency_reward = 1.0
        elif execution_time <= acceptable_time:
            efficiency_reward = 1.0 - (execution_time - optimal_time) / \
                               (acceptable_time - optimal_time) * 0.3
        else:
            penalty = min((execution_time - acceptable_time) / acceptable_time, 1.0)
            efficiency_reward = 0.7 * (1.0 - penalty)
        
        return max(efficiency_reward, 0.0)
    
    def _calculate_coordination_reward(self, result: dict) -> float:
        """Calculate reward based on agent coordination quality"""
        agent_rewards = result.get('agent_rewards', {})
        
        if not agent_rewards:
            return 0.5
        
        rewards = list(agent_rewards.values())
        avg_performance = np.mean(rewards)
        variance = np.var(rewards)
        balance_score = 1.0 / (1.0 + variance * 5)
        min_performance = min(rewards)
        
        coordination_reward = (
            0.5 * avg_performance +
            0.3 * balance_score +
            0.2 * min_performance
        )
        
        return coordination_reward
    
    def _calculate_diversity_reward(self, 
                                    coordination_pattern: int,
                                    pattern_usage: dict = None,
                                    episode: int = 0) -> float:
        """
        Calculate reward for exploration diversity
        
        FIXED: Now gives substantial bonuses for:
        1. Using under-explored patterns (first 60 episodes)
        2. Balanced pattern usage
        3. Trying new patterns
        """
        if pattern_usage is None:
            return 0.5
        
        total_usage = sum(pattern_usage.values())
        if total_usage == 0:
            return 1.0  # First episode bonus
        
        # Phase 1: Exploration bonus (episodes 0-59)
        if episode < 60:
            # Give higher reward for less-used patterns
            pattern_count = pattern_usage[coordination_pattern]
            avg_usage = total_usage / 5.0  # 5 patterns
            
            if pattern_count < avg_usage:
                # Bonus for under-explored patterns
                exploration_bonus = 0.8 + (avg_usage - pattern_count) / avg_usage * 0.2
            else:
                # Penalty for over-explored patterns
                exploration_bonus = 0.5 - (pattern_count - avg_usage) / total_usage * 0.3
            
            return max(0.2, min(1.0, exploration_bonus))
        
        # Phase 2: Balance bonus (episodes 60+)
        else:
            # Calculate entropy of pattern distribution
            pattern_probs = np.array([pattern_usage[i] / total_usage for i in range(5)])
            entropy = -np.sum(pattern_probs * np.log(pattern_probs + 1e-10))
            max_entropy = np.log(5)  # Perfect balance
            
            # Reward balanced usage
            balance_score = entropy / max_entropy
            
            # Small bonus for current pattern if it improves balance
            current_usage_ratio = pattern_usage[coordination_pattern] / total_usage
            ideal_ratio = 0.2  # 1/5 for 5 patterns
            
            if current_usage_ratio < ideal_ratio:
                pattern_bonus = 0.1  # Bonus for using under-represented pattern
            else:
                pattern_bonus = 0.0
            
            diversity_reward = 0.7 * balance_score + 0.3 + pattern_bonus
            
            return min(1.0, diversity_reward)
    
    def _estimate_task_complexity(self, task: dict) -> float:
        """Estimate task complexity (0-1)"""
        requirements = task.get('requirements', {})
        
        length = requirements.get('length', 500)
        length_complexity = min(length / 2000, 1.0)
        
        tone_complexity = {
            'casual': 0.2, 'informative': 0.4, 'persuasive': 0.6,
            'educational': 0.5, 'technical': 0.8, 'academic': 0.9
        }
        tone = requirements.get('tone', 'informative')
        tone_score = tone_complexity.get(tone, 0.5)
        
        complexity = 0.5 * length_complexity + 0.5 * tone_score
        
        return complexity