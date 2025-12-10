"""
Reinforcement Learning for Agentic AI Systems
Main Entry Point - Multi-Agent Content Creation with RL

This system implements:
1. DQN for learning optimal agent coordination
2. Thompson Sampling for exploration in task allocation
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np

from rl_orchestrator import RLAgentOrchestrator
from config import Config
from visualization import Visualizer
from evaluation import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rl_agent_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class RLAgentSystem:
    """Main system orchestrating RL-enhanced agent coordination"""
    
    def __init__(self, config_path: str = None):
        """Initialize the RL Agent System"""
        self.config = Config(config_path) if config_path else Config()
        self.orchestrator = RLAgentOrchestrator(self.config)
        self.visualizer = Visualizer(self.config.output_dir)
        self.evaluator = Evaluator()
        
        # Create output directories
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        logger.info(f"Initialized RL Agent System with config: {self.config}")
    
    def run_training_episode(self, task: dict, episode: int) -> dict:
        """Run a single training episode"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {episode + 1}: {task['type']}")
        logger.info(f"{'='*60}")
        
        # Execute task with RL orchestration
        result = self.orchestrator.execute_task(task)
        
        # Log episode results
        episode_data = {
            'episode': episode + 1,
            'task_type': task['type'],
            'reward': result['reward'],
            'quality_score': result['quality_score'],
            'execution_time': result['execution_time'],
            'agent_sequence': result['agent_sequence'],
            'coordination_pattern': result['coordination_pattern'],
            'exploration_rate': self.orchestrator.dqn_agent.epsilon
        }
        
        logger.info(f"Reward: {result['reward']:.3f}")
        logger.info(f"Quality Score: {result['quality_score']:.3f}")
        logger.info(f"Execution Time: {result['execution_time']:.2f}s")
        logger.info(f"Pattern: {result['coordination_pattern']}")
        logger.info(f"Epsilon: {self.orchestrator.dqn_agent.epsilon:.3f}")
        
        return episode_data
    
    def run_training(self, num_episodes: int = 200):
        """Run the complete training process"""
        logger.info(f"\nStarting training for {num_episodes} episodes...")
        logger.info(f"Forced exploration: Episodes 1-30")
        logger.info(f"Guided exploration: Episodes 31-60")
        logger.info(f"Normal RL: Episodes 61+")
        
        training_history = []
        
        # Define diverse tasks for training
        tasks = self._generate_training_tasks()
        
        for episode in range(num_episodes):
            # Select task (rotate through different types)
            task = tasks[episode % len(tasks)]
            
            # Run episode
            episode_data = self.run_training_episode(task, episode)
            training_history.append(episode_data)
            
            # Log pattern statistics every 30 episodes
            if (episode + 1) % 30 == 0:
                self._log_pattern_stats()
            
            # Save checkpoint every 10 episodes
            if (episode + 1) % 10 == 0:
                self._save_checkpoint(episode + 1, training_history)
                self._plot_progress(training_history)
        
        # Final evaluation
        logger.info("\n" + "="*60)
        logger.info("Training Complete - Running Final Evaluation")
        logger.info("="*60)
        
        self._run_final_evaluation(training_history)
        
        return training_history
    
    def _log_pattern_stats(self):
        """Log pattern usage statistics"""
        stats = self.orchestrator.get_pattern_statistics()
        
        logger.info("\n" + "-"*60)
        logger.info("PATTERN USAGE STATISTICS")
        logger.info("-"*60)
        
        for pattern_name, pattern_stats in stats.items():
            logger.info(f"{pattern_name}:")
            logger.info(f"  Usage: {pattern_stats['usage_count']} times ({pattern_stats['usage_percentage']:.1f}%)")
            logger.info(f"  Avg Reward: {pattern_stats['avg_reward']:.3f}")
        
        logger.info("-"*60)
    
    def _generate_training_tasks(self) -> list:
        """Generate diverse training tasks"""
        tasks = [
            {
                'type': 'blog_post',
                'topic': 'The Future of Artificial Intelligence',
                'requirements': {
                    'length': 800,
                    'tone': 'informative',
                    'target_audience': 'general'
                }
            },
            {
                'type': 'technical_article',
                'topic': 'Reinforcement Learning in Production Systems',
                'requirements': {
                    'length': 1200,
                    'tone': 'technical',
                    'target_audience': 'developers'
                }
            },
            {
                'type': 'marketing_copy',
                'topic': 'Revolutionary AI-Powered Content Platform',
                'requirements': {
                    'length': 500,
                    'tone': 'persuasive',
                    'target_audience': 'business'
                }
            },
            {
                'type': 'research_summary',
                'topic': 'Recent Advances in Multi-Agent Systems',
                'requirements': {
                    'length': 1000,
                    'tone': 'academic',
                    'target_audience': 'researchers'
                }
            },
            {
                'type': 'tutorial',
                'topic': 'Getting Started with Reinforcement Learning',
                'requirements': {
                    'length': 1500,
                    'tone': 'educational',
                    'target_audience': 'students'
                }
            }
        ]
        return tasks
    
    def _save_checkpoint(self, episode: int, history: list):
        """Save training checkpoint"""
        checkpoint = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'history': history,
            'dqn_state': self.orchestrator.dqn_agent.get_state(),
            'thompson_state': self.orchestrator.thompson_sampler.get_state(),
            'pattern_stats': self.orchestrator.get_pattern_statistics()
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_ep{episode}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, cls=NumpyEncoder)
        
        # Save models
        self.orchestrator.save_models(self.output_dir / "models" / f"model_ep{episode}")
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _plot_progress(self, history: list):
        """Plot training progress"""
        self.visualizer.plot_training_curves(history)
        self.visualizer.plot_agent_utilization(history)
        self.visualizer.plot_exploration_vs_exploitation(history)
        logger.info("Progress plots updated")
    
    def _run_final_evaluation(self, history: list):
        """Run comprehensive final evaluation"""
        # Calculate metrics
        metrics = self.evaluator.calculate_metrics(history)
        
        # Add pattern statistics
        metrics['pattern_statistics'] = self.orchestrator.get_pattern_statistics()
        
        # Generate visualizations
        self.visualizer.plot_final_results(history, metrics)
        
        # Save evaluation report
        report_path = self.output_dir / "results" / "final_evaluation.json"
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("FINAL EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Average Reward: {metrics['avg_reward']:.3f}")
        logger.info(f"Average Quality: {metrics['avg_quality']:.3f}")
        logger.info(f"Improvement: {metrics['improvement_percentage']:.1f}%")
        logger.info(f"Convergence Episode: {metrics.get('convergence_episode', 'N/A')}")
        logger.info(f"Best Episode: {metrics['best_episode']}")
        
        # Print pattern statistics
        logger.info("\n" + "-"*60)
        logger.info("FINAL PATTERN USAGE")
        logger.info("-"*60)
        for pattern_name, pattern_stats in metrics['pattern_statistics'].items():
            logger.info(f"{pattern_name}: {pattern_stats['usage_count']} times ({pattern_stats['usage_percentage']:.1f}%) - Avg Reward: {pattern_stats['avg_reward']:.3f}")
        
        logger.info(f"\nReport saved: {report_path}")


def main():
    """Main execution function"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  RL-Enhanced Multi-Agent Content Creation System          ║
    ║  Final Project: Reinforcement Learning for Agentic AI     ║
    ║  FIXED VERSION - Pattern Diversity Enabled                ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize system
    system = RLAgentSystem()
    
    # Run training
    num_episodes = int(input("\nEnter number of training episodes: ") or "200")
    
    try:
        history = system.run_training(num_episodes)
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print(f"Results saved to: {system.output_dir}")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        print("\nTraining interrupted. Partial results saved.")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()