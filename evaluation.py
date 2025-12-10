"""
Evaluation Module for RL Agent System Performance

Calculates comprehensive metrics, convergence analysis,
and statistical significance tests.
"""

import numpy as np
from scipy import stats
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates RL agent system performance"""
    
    def __init__(self):
        self.baseline_reward = 0.5
        self.baseline_quality = 0.5
    
    def calculate_metrics(self, history: List[dict]) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary of evaluation metrics
        """
        
        rewards = [h['reward'] for h in history]
        qualities = [h['quality_score'] for h in history]
        execution_times = [h['execution_time'] for h in history]
        
        metrics = {
            # Reward metrics
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'best_reward': np.max(rewards),
            'best_episode': int(np.argmax(rewards) + 1),
            
            # Quality metrics
            'avg_quality': np.mean(qualities),
            'std_quality': np.std(qualities),
            'min_quality': np.min(qualities),
            'max_quality': np.max(qualities),
            'best_quality': np.max(qualities),
            
            # Efficiency metrics
            'avg_execution_time': np.mean(execution_times),
            'std_execution_time': np.std(execution_times),
            
            # Learning progress
            'initial_avg_reward': np.mean(rewards[:10]) if len(rewards) >= 10 else np.mean(rewards),
            'final_avg_reward': np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards),
            'initial_avg_quality': np.mean(qualities[:10]) if len(qualities) >= 10 else np.mean(qualities),
            'final_avg_quality': np.mean(qualities[-10:]) if len(qualities) >= 10 else np.mean(qualities),
        }
        
        # Calculate improvement
        if metrics['initial_avg_reward'] > 0:
            metrics['improvement_percentage'] = (
                (metrics['final_avg_reward'] - metrics['initial_avg_reward']) / 
                metrics['initial_avg_reward'] * 100
            )
        else:
            metrics['improvement_percentage'] = 0.0
        
        # Convergence analysis
        convergence_results = self._analyze_convergence(rewards, qualities)
        metrics.update(convergence_results)
        
        # Statistical tests
        statistical_results = self._statistical_tests(rewards, qualities)
        metrics.update(statistical_results)
        
        # Stability metrics
        stability_results = self._analyze_stability(rewards[-20:] if len(rewards) >= 20 else rewards)
        metrics.update(stability_results)
        
        logger.info(f"Calculated {len(metrics)} performance metrics")
        
        return metrics
    
    def _analyze_convergence(self, rewards: List[float], qualities: List[float]) -> Dict:
        """
        Analyze convergence of learning
        
        Returns:
            Convergence metrics
        """
        
        # Simple convergence detection: when moving average stabilizes
        window = 10
        threshold = 0.05  # 5% variation threshold
        
        convergence_episode = None
        
        if len(rewards) >= window * 2:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            
            for i in range(len(moving_avg) - window):
                recent_avg = np.mean(moving_avg[i:i+window])
                previous_avg = np.mean(moving_avg[max(0, i-window):i])
                
                if previous_avg > 0:
                    change = abs(recent_avg - previous_avg) / previous_avg
                    
                    if change < threshold:
                        convergence_episode = i + window
                        break
        
        return {
            'convergence_episode': int(convergence_episode) if convergence_episode else None,
            'converged': bool(convergence_episode is not None)
        }
    
    def _statistical_tests(self, rewards: List[float], qualities: List[float]) -> Dict:
        """
        Perform statistical significance tests
        
        Tests if final performance is significantly better than initial performance
        """
        
        if len(rewards) < 20:
            return {
                'reward_improvement_significant': False,
                'quality_improvement_significant': False,
                'reward_pvalue': 1.0,
                'quality_pvalue': 1.0
            }
        
        # Split into initial and final periods
        initial_rewards = rewards[:10]
        final_rewards = rewards[-10:]
        
        initial_qualities = qualities[:10]
        final_qualities = qualities[-10:]
        
        # Perform t-tests
        reward_ttest = stats.ttest_ind(final_rewards, initial_rewards, alternative='greater')
        quality_ttest = stats.ttest_ind(final_qualities, initial_qualities, alternative='greater')
        
        significance_level = 0.05
        
        return {
            'reward_improvement_significant': bool(reward_ttest.pvalue < significance_level),
            'quality_improvement_significant': bool(quality_ttest.pvalue < significance_level),
            'reward_pvalue': float(reward_ttest.pvalue),
            'quality_pvalue': float(quality_ttest.pvalue),
            'reward_tstatistic': float(reward_ttest.statistic),
            'quality_tstatistic': float(quality_ttest.statistic)
        }
    
    def _analyze_stability(self, recent_rewards: List[float]) -> Dict:
        """
        Analyze stability of recent performance
        
        Measures consistency and reliability of learned policy
        """
        
        if not recent_rewards:
            return {
                'stability_score': 0.0,
                'coefficient_of_variation': 0.0
            }
        
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        
        # Coefficient of variation (lower is more stable)
        cv = (std_reward / mean_reward) if mean_reward > 0 else float('inf')
        
        # Stability score (0-1, higher is better)
        # Based on inverse of coefficient of variation
        stability_score = 1.0 / (1.0 + cv) if cv != float('inf') else 0.0
        
        return {
            'stability_score': float(stability_score),
            'coefficient_of_variation': float(cv)
        }
    
    def compare_to_baseline(self, history: List[dict]) -> Dict:
        """
        Compare learned performance to baseline
        
        Returns:
            Comparison metrics
        """
        
        rewards = [h['reward'] for h in history]
        qualities = [h['quality_score'] for h in history]
        
        final_reward = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
        final_quality = np.mean(qualities[-10:]) if len(qualities) >= 10 else np.mean(qualities)
        
        reward_improvement = ((final_reward - self.baseline_reward) / 
                             self.baseline_reward * 100)
        quality_improvement = ((final_quality - self.baseline_quality) / 
                              self.baseline_quality * 100)
        
        return {
            'baseline_reward': self.baseline_reward,
            'learned_reward': float(final_reward),
            'reward_improvement_vs_baseline': float(reward_improvement),
            'baseline_quality': self.baseline_quality,
            'learned_quality': float(final_quality),
            'quality_improvement_vs_baseline': float(quality_improvement),
            'exceeds_baseline': bool(final_reward > self.baseline_reward)  # Explicit bool conversion
        }
    
    def analyze_exploration_efficiency(self, history: List[dict]) -> Dict:
        """
        Analyze efficiency of exploration strategy
        
        Measures how effectively the agent explored the action space
        """
        
        coordination_patterns = [h.get('coordination_pattern', 0) for h in history]  # Use .get() for safety
        unique_patterns = set(coordination_patterns)
        
        # Pattern diversity score
        pattern_diversity = len(unique_patterns) / 5.0  # 5 total patterns
        
        # Pattern usage distribution (entropy)
        pattern_counts = np.array([coordination_patterns.count(i) for i in range(5)])
        pattern_probs = pattern_counts / len(coordination_patterns) if len(coordination_patterns) > 0 else np.zeros(5)
        
        # Calculate entropy (higher entropy = more diverse exploration)
        entropy = -np.sum(pattern_probs * np.log(pattern_probs + 1e-10))
        max_entropy = np.log(5)  # Maximum entropy for 5 patterns
        normalized_entropy = entropy / max_entropy
        
        # Exploration phases
        early_patterns = set(coordination_patterns[:len(coordination_patterns)//3]) if len(coordination_patterns) >= 3 else set(coordination_patterns)
        late_patterns = set(coordination_patterns[-len(coordination_patterns)//3:]) if len(coordination_patterns) >= 3 else set(coordination_patterns)
        
        exploration_phases = {
            'early_diversity': len(early_patterns) / 5.0,
            'late_diversity': len(late_patterns) / 5.0,
            'exploration_reduction': (len(early_patterns) - len(late_patterns))
        }
        
        return {
            'pattern_diversity': float(pattern_diversity),
            'exploration_entropy': float(normalized_entropy),
            'unique_patterns_explored': len(unique_patterns),
            **exploration_phases
        }
    
    def generate_report(self, history: List[dict]) -> str:
        """
        Generate comprehensive evaluation report
        
        Returns:
            Formatted text report
        """
        
        metrics = self.calculate_metrics(history)
        baseline_comparison = self.compare_to_baseline(history)
        exploration_analysis = self.analyze_exploration_efficiency(history)
        
        report = f"""
╔═══════════════════════════════════════════════════════════════════╗
║         REINFORCEMENT LEARNING AGENT SYSTEM EVALUATION            ║
╚═══════════════════════════════════════════════════════════════════╝

TRAINING SUMMARY
{'='*70}
Total Episodes: {len(history)}
Convergence: {metrics['convergence_episode']}
Best Episode: {metrics['best_episode']}

PERFORMANCE METRICS
{'='*70}
Reward Performance:
  • Average: {metrics['avg_reward']:.4f} ± {metrics['std_reward']:.4f}
  • Best: {metrics['best_reward']:.4f}
  • Initial (first 10): {metrics['initial_avg_reward']:.4f}
  • Final (last 10): {metrics['final_avg_reward']:.4f}
  • Improvement: {metrics['improvement_percentage']:.2f}%

Quality Performance:
  • Average: {metrics['avg_quality']:.4f} ± {metrics['std_quality']:.4f}
  • Best: {metrics['best_quality']:.4f}
  • Initial (first 10): {metrics['initial_avg_quality']:.4f}
  • Final (last 10): {metrics['final_avg_quality']:.4f}

Efficiency:
  • Average Execution Time: {metrics['avg_execution_time']:.3f}s ± {metrics['std_execution_time']:.3f}s

BASELINE COMPARISON
{'='*70}
Reward:
  • Baseline: {baseline_comparison['baseline_reward']:.4f}
  • Learned: {baseline_comparison['learned_reward']:.4f}
  • Improvement: {baseline_comparison['reward_improvement_vs_baseline']:.2f}%
  • Exceeds Baseline: {'✓' if baseline_comparison['exceeds_baseline'] else '✗'}

Quality:
  • Baseline: {baseline_comparison['baseline_quality']:.4f}
  • Learned: {baseline_comparison['learned_quality']:.4f}
  • Improvement: {baseline_comparison['quality_improvement_vs_baseline']:.2f}%

STATISTICAL SIGNIFICANCE
{'='*70}
Reward Improvement:
  • Significant: {'✓' if metrics['reward_improvement_significant'] else '✗'}
  • p-value: {metrics['reward_pvalue']:.4f}
  • t-statistic: {metrics['reward_tstatistic']:.4f}

Quality Improvement:
  • Significant: {'✓' if metrics['quality_improvement_significant'] else '✗'}
  • p-value: {metrics['quality_pvalue']:.4f}
  • t-statistic: {metrics['quality_tstatistic']:.4f}

STABILITY ANALYSIS
{'='*70}
  • Stability Score: {metrics['stability_score']:.4f}
  • Coefficient of Variation: {metrics['coefficient_of_variation']:.4f}

EXPLORATION EFFICIENCY
{'='*70}
  • Pattern Diversity: {exploration_analysis['pattern_diversity']:.4f}
  • Exploration Entropy: {exploration_analysis['exploration_entropy']:.4f}
  • Unique Patterns: {exploration_analysis['unique_patterns_explored']}/5
  • Early Diversity: {exploration_analysis['early_diversity']:.4f}
  • Late Diversity: {exploration_analysis['late_diversity']:.4f}

{'='*70}
Report Generated: {len(history)} episodes analyzed
{'='*70}
"""
        
        return report