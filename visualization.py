"""
Visualization Module for RL Training Results

Creates comprehensive visualizations of learning curves,
agent performance, and coordination patterns.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict
import seaborn as sns
import textwrap

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class Visualizer:
    """Creates visualizations for RL training"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Visualizer initialized with output dir: {self.plots_dir}")
    
    def plot_training_curves(self, history: List[dict]):
        """Plot reward and quality learning curves"""
        
        episodes = [h['episode'] for h in history]
        rewards = [h['reward'] for h in history]
        qualities = [h['quality_score'] for h in history]
        
        # Calculate moving averages
        window = 10
        rewards_ma = self._moving_average(rewards, window)
        qualities_ma = self._moving_average(qualities, window)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Rewards plot
        ax1.plot(episodes, rewards, alpha=0.3, label='Episode Reward')
        ax1.plot(episodes[window-1:], rewards_ma, linewidth=2, 
                label=f'{window}-Episode Moving Average')
        ax1.axhline(y=np.mean(rewards), color='r', linestyle='--', 
                label=f'Mean: {np.mean(rewards):.3f}')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Learning Curve: Reward over Episodes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Quality plot
        ax2.plot(episodes, qualities, alpha=0.3, label='Quality Score', color='green')
        ax2.plot(episodes[window-1:], qualities_ma, linewidth=2, 
                label=f'{window}-Episode Moving Average', color='darkgreen')
        ax2.axhline(y=np.mean(qualities), color='r', linestyle='--',
                label=f'Mean: {np.mean(qualities):.3f}')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Quality Score')
        ax2.set_title('Learning Curve: Quality over Episodes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training curves saved")
    
    def plot_agent_utilization(self, history: List[dict]):
        """Plot agent utilization and performance"""
        
        # Count agent usage
        agent_usage = {}
        agent_quality = {}
        
        for h in history:
            for agent_id in h['agent_sequence']:
                agent_usage[agent_id] = agent_usage.get(agent_id, 0) + 1
                if agent_id not in agent_quality:
                    agent_quality[agent_id] = []
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Utilization
        agents = sorted(agent_usage.keys())
        usage_counts = [agent_usage[a] for a in agents]
        
        ax1.bar(agents, usage_counts, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Agent ID')
        ax1.set_ylabel('Number of Times Used')
        ax1.set_title('Agent Utilization Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Utilization percentage pie chart
        labels = [f'Agent {a}' for a in agents]
        ax2.pie(usage_counts, labels=labels, autopct='%1.1f%%',
            colors=sns.color_palette('Set2', len(agents)))
        ax2.set_title('Agent Utilization Distribution')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'agent_utilization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Agent utilization plot saved")
    
    def plot_exploration_vs_exploitation(self, history: List[dict]):
        """Plot exploration rate over time"""
        
        episodes = [h['episode'] for h in history]
        exploration_rates = [h['exploration_rate'] for h in history]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(episodes, exploration_rates, linewidth=2, color='purple')
        ax.fill_between(episodes, 0, exploration_rates, alpha=0.3, color='purple')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Exploration Rate (ε)')
        ax.set_title('Exploration vs Exploitation: Epsilon Decay')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Add annotations
        if exploration_rates:
            ax.annotate(f'Start: ε={exploration_rates[0]:.3f}',
                        xy=(episodes[0], exploration_rates[0]),
                        xytext=(20, -30), textcoords='offset points',
                        bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='black'))
            
            ax.annotate(f'End: ε={exploration_rates[-1]:.3f}',
                        xy=(episodes[-1], exploration_rates[-1]),
                        xytext=(-100, 30), textcoords='offset points',
                        bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='black'))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'exploration_rate.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Exploration rate plot saved")
    
    def plot_final_results(self, history: List[dict], metrics: Dict):
        """Create comprehensive final results visualization"""
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        episodes = [h['episode'] for h in history]
        rewards = [h['reward'] for h in history]
        qualities = [h['quality_score'] for h in history]
        execution_times = [h['execution_time'] for h in history]
        
        # 1. Reward progression
        ax1 = fig.add_subplot(gs[0, 0])
        window = 10
        rewards_ma = self._moving_average(rewards, window)
        ax1.plot(episodes, rewards, alpha=0.2, color='blue')
        ax1.plot(episodes[window-1:], rewards_ma, linewidth=2, 
                color='darkblue', label='Moving Average')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Reward Learning Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Quality distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(qualities, bins=20, color='green', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(qualities), color='red', linestyle='--',
                label=f'Mean: {np.mean(qualities):.3f}')
        ax2.set_xlabel('Quality Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Quality Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Execution time
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(episodes, execution_times, alpha=0.5, color='orange')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Execution Time per Episode')
        ax3.grid(True, alpha=0.3)
        
        # 4. Coordination patterns
        ax4 = fig.add_subplot(gs[1, 1])
        patterns = [h.get('coordination_pattern', 0) for h in history]  # Use .get() for safety
        pattern_counts = {i: patterns.count(i) for i in set(patterns)}
        pattern_names = ['Sequential', 'Parallel', 'Hierarchical', 
                        'Collaborative', 'Adaptive']
        
        bars = ax4.bar(range(len(pattern_counts)), 
                        [pattern_counts.get(i, 0) for i in range(5)],
                        color='coral', alpha=0.7)
        ax4.set_xticks(range(5))
        ax4.set_xticklabels(pattern_names, rotation=45, ha='right')
        ax4.set_ylabel('Usage Count')
        ax4.set_title('Coordination Pattern Usage')
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance metrics summary
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        
        summary_text = textwrap.dedent( f"""
        FINAL TRAINING SUMMARY
        {'='*45}
        
        Total Episodes: {len(history)}
        
        Performance Metrics:
        • Average Reward: {metrics['avg_reward']:.4f}
        • Best Reward: {metrics['best_reward']:.4f}
        • Final 10-Episode Average: {metrics['final_avg_reward']:.4f}
        
        • Average Quality: {metrics['avg_quality']:.4f}
        • Best Quality: {metrics['best_quality']:.4f}
        
        Learning Progress:
        • Improvement: {metrics['improvement_percentage']:.2f}%
        • Convergence Episode: {metrics.get('convergence_episode', 'N/A')}
        • Best Episode: {metrics['best_episode']}
        
        Efficiency:
        • Average Execution Time: {metrics['avg_execution_time']:.2f}s
        • Total Training Time: {metrics.get('total_training_time', 'N/A')}
        """)
        
        ax5.text(0.04, 0.36, summary_text, fontsize=11, family='monospace', verticalalignment='center')
        
        plt.savefig(self.plots_dir / 'final_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Final results visualization saved")
    
    def _moving_average(self, data: List[float], window: int) -> np.ndarray:
        """Calculate moving average"""
        if len(data) < window:
            return np.array(data)
        
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def plot_coordination_heatmap(self, history: List[dict]):
        """Plot heatmap of task type vs coordination pattern"""
        
        # Create matrix
        task_types = ['blog_post', 'technical_article', 'marketing_copy',
                    'research_summary', 'tutorial']
        patterns = ['Sequential', 'Parallel', 'Hierarchical', 
                    'Collaborative', 'Adaptive']
        
        matrix = np.zeros((len(task_types), len(patterns)))
        
        for h in history:
            task_type = h.get('task_type', 'blog_post')  # Use .get() for safety
            pattern_id = h.get('coordination_pattern', 0)  # Use .get() for safety
            
            task_idx = task_types.index(task_type) if task_type in task_types else 0
            pattern_idx = pattern_id
            matrix[task_idx][pattern_idx] += 1
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(patterns)))
        ax.set_yticks(np.arange(len(task_types)))
        ax.set_xticklabels(patterns, rotation=45, ha='right')
        ax.set_yticklabels(task_types)
        
        # Add text annotations
        for i in range(len(task_types)):
            for j in range(len(patterns)):
                text = ax.text(j, i, int(matrix[i, j]),
                            ha="center", va="center", color="black")
        
        ax.set_title('Task Type vs Coordination Pattern Usage')
        plt.colorbar(im, ax=ax, label='Usage Count')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'coordination_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Coordination heatmap saved")