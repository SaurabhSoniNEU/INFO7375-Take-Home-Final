"""
Configuration Management for RL Agent System
"""

import json
import os
from pathlib import Path


class Config:
    """Configuration for RL-Enhanced Agent System"""
    
    def __init__(self, config_path: str = None):
        # Default configuration - OPTIMIZED
        self.config = {
            # System settings
            'output_dir': 'output',
            'log_level': 'INFO',
            'seed': 42,
            
            # Agent settings
            'num_agents': 4,
            'num_coordination_patterns': 5,
            
            # DQN settings - OPTIMIZED
            'dqn_learning_rate': 0.001,
            'dqn_gamma': 0.95,
            'dqn_epsilon_start': 1.0,
            'dqn_epsilon_end': 0.01,
            'dqn_epsilon_decay': 0.97,  # FIXED: More aggressive decay
            'dqn_buffer_size': 10000,
            'dqn_batch_size': 64,
            'dqn_target_update_freq': 10,
            'dqn_hidden_dims': [256, 128, 64],
            
            # Thompson Sampling settings
            'thompson_alpha_prior': 1.0,
            'thompson_beta_prior': 1.0,
            'thompson_weight': 0.5,
            
            # UCB settings
            'ucb_exploration_param': 2.0,
            
            # Reward settings - OPTIMIZED
            'reward_quality_weight': 0.4,  # Reduced from 0.5
            'reward_efficiency_weight': 0.2,
            'reward_coordination_weight': 0.2,
            'reward_diversity_weight': 0.2,  # Increased from 0.1
            
            # Training settings
            'num_episodes': 200,  # Increased default
            'checkpoint_frequency': 10,
            'evaluation_frequency': 10,
            
            # Task settings
            'task_types': ['blog_post', 'technical_article', 'marketing_copy',
                          'research_summary', 'tutorial']
        }
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
            self.config.update(loaded_config)
    
    def save_config(self, config_path: str):
        """Save configuration to JSON file"""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def __getattr__(self, name):
        """Allow attribute-style access to config values"""
        if name == 'config':
            return object.__getattribute__(self, 'config')
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __repr__(self):
        return f"Config({json.dumps(self.config, indent=2)})"


def create_default_config(output_path: str = "config.json"):
    """Create and save default configuration file"""
    config = Config()
    config.save_config(output_path)
    print(f"Default configuration saved to {output_path}")


if __name__ == "__main__":
    create_default_config()