"""
System Test and Verification Script

Verifies that all components are working correctly
and runs a quick test episode.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported"""
    print("\n" + "="*60)
    print("Testing Imports...")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        from config import Config
        print("✓ Config module")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from dqn_agent import DQNAgent
        print("✓ DQN Agent module")
    except ImportError as e:
        print(f"✗ DQN Agent import failed: {e}")
        return False
    
    try:
        from thompson_sampler import ThompsonSampler
        print("✓ Thompson Sampler module")
    except ImportError as e:
        print(f"✗ Thompson Sampler import failed: {e}")
        return False
    
    try:
        from agents import AgentTeam
        print("✓ Agents module")
    except ImportError as e:
        print(f"✗ Agents import failed: {e}")
        return False
    
    try:
        from reward_function import RewardCalculator
        print("✓ Reward Function module")
    except ImportError as e:
        print(f"✗ Reward Function import failed: {e}")
        return False
    
    try:
        from rl_orchestrator import RLAgentOrchestrator
        print("✓ RL Orchestrator module")
    except ImportError as e:
        print(f"✗ RL Orchestrator import failed: {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True


def test_dqn_agent():
    """Test DQN agent initialization and basic operations"""
    print("\n" + "="*60)
    print("Testing DQN Agent...")
    print("="*60)
    
    try:
        from dqn_agent import DQNAgent
        import numpy as np
        
        agent = DQNAgent(state_dim=10, action_dim=5)
        
        # Test action selection
        state = np.random.randn(10)
        action = agent.select_action(state)
        assert 0 <= action < 5, "Invalid action"
        print(f"✓ Action selection: {action}")
        
        # Test experience storage
        next_state = np.random.randn(10)
        agent.store_experience(state, action, 0.5, next_state, False)
        print(f"✓ Experience storage")
        
        # Test Q-values
        q_values = agent.get_q_values(state)
        assert len(q_values) == 5, "Invalid Q-values shape"
        print(f"✓ Q-value calculation: {q_values}")
        
        print("\n✓ DQN Agent tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ DQN Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_thompson_sampler():
    """Test Thompson Sampler"""
    print("\n" + "="*60)
    print("Testing Thompson Sampler...")
    print("="*60)
    
    try:
        from thompson_sampler import ThompsonSampler
        
        sampler = ThompsonSampler(num_agents=4)
        
        # Test sampling
        agent_id = sampler.sample_agent()
        assert 0 <= agent_id < 4, "Invalid agent ID"
        print(f"✓ Agent sampling: {agent_id}")
        
        # Test update
        sampler.update(agent_id, reward=0.7, success=True)
        print(f"✓ Update with reward")
        
        # Test stats
        stats = sampler.get_agent_stats(agent_id)
        print(f"✓ Agent stats: {stats}")
        
        print("\n✓ Thompson Sampler tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Thompson Sampler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agents():
    """Test agent team"""
    print("\n" + "="*60)
    print("Testing Agent Team...")
    print("="*60)
    
    try:
        from agents import AgentTeam
        from config import Config
        
        config = Config()
        team = AgentTeam(config)
        
        # Test task
        task = {
            'type': 'blog_post',
            'topic': 'Test Topic',
            'requirements': {
                'length': 500,
                'tone': 'informative'
            }
        }
        
        # Test agent execution
        agent = team.agents[0]
        result = agent.execute(task)
        
        assert 'content' in result, "Missing content"
        assert 'quality' in result, "Missing quality"
        print(f"✓ Agent execution: quality={result['quality']:.3f}")
        
        print("\n✓ Agent Team tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Agent Team test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_calculator():
    """Test reward calculation"""
    print("\n" + "="*60)
    print("Testing Reward Calculator...")
    print("="*60)
    
    try:
        from reward_function import RewardCalculator
        
        calculator = RewardCalculator()
        
        task = {'type': 'blog_post', 'requirements': {}}
        result = {'quality_score': 0.8, 'agent_rewards': {0: 0.8, 1: 0.75}}
        
        reward = calculator.calculate_reward(task, result, 5.0, 0)
        
        assert 0 <= reward <= 1, "Invalid reward value"
        print(f"✓ Reward calculation: {reward:.3f}")
        
        print("\n✓ Reward Calculator tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Reward Calculator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_episode():
    """Test a complete training episode"""
    print("\n" + "="*60)
    print("Testing Full Episode...")
    print("="*60)
    
    try:
        from rl_orchestrator import RLAgentOrchestrator
        from config import Config
        
        config = Config()
        orchestrator = RLAgentOrchestrator(config)
        
        task = {
            'type': 'blog_post',
            'topic': 'Test Article',
            'requirements': {
                'length': 500,
                'tone': 'informative',
                'target_audience': 'general'
            }
        }
        
        result = orchestrator.execute_task(task)
        
        assert 'reward' in result, "Missing reward"
        assert 'quality_score' in result, "Missing quality score"
        assert 'execution_time' in result, "Missing execution time"
        
        print(f"✓ Task execution completed:")
        print(f"  - Reward: {result['reward']:.3f}")
        print(f"  - Quality: {result['quality_score']:.3f}")
        print(f"  - Time: {result['execution_time']:.3f}s")
        print(f"  - Pattern: {result['coordination_pattern']}")
        
        print("\n✓ Full episode test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Full episode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           RL Agent System - Test Suite                   ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    tests = [
        ("Imports", test_imports),
        ("DQN Agent", test_dqn_agent),
        ("Thompson Sampler", test_thompson_sampler),
        ("Agent Team", test_agents),
        ("Reward Calculator", test_reward_calculator),
        ("Full Episode", test_full_episode),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nYour system is ready to use.")
        print("Run 'python main.py' to start training.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the failing tests before running the main system.")
        print("Check that all dependencies are installed correctly.")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)