"""
Specialized Agent Team for Content Creation

Implements individual agents with specific capabilities
that collaborate under RL orchestration.
"""

import random
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for specialized agents"""
    
    def __init__(self, agent_id: int, name: str, specialization: str):
        self.agent_id = agent_id
        self.name = name
        self.specialization = specialization
        self.performance_baseline = 0.5 + random.uniform(-0.1, 0.1)
    
    def execute(self, task: dict, previous_content: str = "") -> dict:
        """Execute task (to be overridden by subclasses)"""
        raise NotImplementedError
    
    def _calculate_quality(self, task: dict, content: str) -> float:
        """Calculate quality score for generated content"""
        # Simulated quality based on task alignment and agent specialization
        
        task_type = task['type']
        base_quality = self.performance_baseline
        
        # Specialization bonus
        if self._is_specialized_for(task_type):
            base_quality += 0.2
        
        # Requirements alignment
        requirements = task.get('requirements', {})
        if self._meets_requirements(requirements):
            base_quality += 0.1
        
        # Content length bonus
        target_length = requirements.get('length', 500)
        if content:  # Check if content exists
            actual_length = len(content.split())
            length_ratio = min(actual_length / target_length, 1.0)
            base_quality += 0.1 * length_ratio
        
        # Add some noise
        quality = base_quality + random.uniform(-0.05, 0.05)
        
        return min(max(quality, 0.0), 1.0)
    
    def _is_specialized_for(self, task_type: str) -> bool:
        """Check if agent is specialized for task type"""
        return task_type in self.get_specializations()
    
    def _meets_requirements(self, requirements: dict) -> bool:
        """Check if agent can meet task requirements"""
        return True  # Simplified
    
    def get_specializations(self) -> List[str]:
        """Get list of task types this agent specializes in"""
        return []
    
    def __repr__(self):
        return f"{self.name} (ID: {self.agent_id})"


class ResearchAgent(BaseAgent):
    """Agent specialized in research and information gathering"""
    
    def __init__(self, agent_id: int):
        super().__init__(agent_id, "Research Agent", "research")
        self.performance_baseline = 0.6
    
    def execute(self, task: dict, previous_content: str = "") -> dict:
        logger.debug(f"{self.name} executing research task")
        
        # Simulate research process
        topic = task.get('topic', 'Unknown')
        requirements = task.get('requirements', {})
        
        # Generate content
        content = self._generate_research_content(topic, requirements, previous_content)
        
        # Calculate quality
        quality = self._calculate_quality(task, content)
        
        return {
            'content': content,
            'quality': quality,
            'agent': self.name
        }
    
    def _generate_research_content(self, topic: str, requirements: dict, previous: str) -> str:
        """Generate research-focused content"""
        if previous:
            return f"{previous}\n\n[Research Analysis by {self.name}]\n" \
                f"Based on comprehensive research on {topic}, " \
                f"the following insights emerge: {self._get_insights()}"
        else:
            return f"[Research Report by {self.name}]\n" \
                f"Topic: {topic}\n\n" \
                f"Research findings: {self._get_insights()}"
    
    def _get_insights(self) -> str:
        insights = [
            "Recent studies indicate significant developments in this area.",
            "Multiple sources corroborate these findings with strong evidence.",
            "The data suggests a clear trend toward innovation and growth.",
            "Expert consensus points to these key factors as critical."
        ]
        return random.choice(insights)
    
    def get_specializations(self) -> List[str]:
        return ['research_summary', 'technical_article', 'tutorial']


class WritingAgent(BaseAgent):
    """Agent specialized in creative and persuasive writing"""
    
    def __init__(self, agent_id: int):
        super().__init__(agent_id, "Writing Agent", "writing")
        self.performance_baseline = 0.65
    
    def execute(self, task: dict, previous_content: str = "") -> dict:
        logger.debug(f"{self.name} executing writing task")
        
        topic = task.get('topic', 'Unknown')
        requirements = task.get('requirements', {})
        
        content = self._generate_written_content(topic, requirements, previous_content)
        quality = self._calculate_quality(task, content)
        
        return {
            'content': content,
            'quality': quality,
            'agent': self.name
        }
    
    def _generate_written_content(self, topic: str, requirements: dict, previous: str) -> str:
        tone = requirements.get('tone', 'informative')
        
        if previous:
            return f"{previous}\n\n[Enhanced by {self.name}]\n" \
                    f"With a {tone} tone: {self._craft_message(topic)}"
        else:
            return f"[Content by {self.name}]\n" \
                f"{self._craft_opening(topic, tone)}\n\n" \
                f"{self._craft_message(topic)}"
    
    def _craft_opening(self, topic: str, tone: str) -> str:
        openings = {
            'informative': f"Let's explore {topic} in detail.",
            'persuasive': f"Discover why {topic} matters now more than ever.",
            'educational': f"Understanding {topic} is crucial for your success.",
            'technical': f"A deep dive into {topic} and its implications."
        }
        return openings.get(tone, f"About {topic}")
    
    def _craft_message(self, topic: str) -> str:
        messages = [
            f"The landscape of {topic} is rapidly evolving, presenting new opportunities.",
            f"Understanding {topic} requires careful consideration of multiple factors.",
            f"The future of {topic} depends on how we approach these challenges today.",
            f"Innovation in {topic} continues to drive transformation across industries."
        ]
        return random.choice(messages)
    
    def get_specializations(self) -> List[str]:
        return ['blog_post', 'marketing_copy', 'tutorial']


class EditorAgent(BaseAgent):
    """Agent specialized in editing and quality improvement"""
    
    def __init__(self, agent_id: int):
        super().__init__(agent_id, "Editor Agent", "editing")
        self.performance_baseline = 0.7
    
    def execute(self, task: dict, previous_content: str = "") -> dict:
        logger.debug(f"{self.name} executing editing task")
        
        if not previous_content:
            # Editor needs content to edit
            content = "[Awaiting content to edit]"
            quality = 0.3
        else:
            content = self._edit_content(previous_content, task)
            quality = self._calculate_quality(task, content)
            quality += 0.1  # Editors typically improve quality
        
        return {
            'content': content,
            'quality': min(quality, 1.0),
            'agent': self.name
        }
    
    def _edit_content(self, content: str, task: dict) -> str:
        """Edit and improve content"""
        return f"{content}\n\n[Refined by {self.name}]\n" \
                f"[Quality improvements: structure, clarity, and flow enhanced]"
    
    def get_specializations(self) -> List[str]:
        return ['blog_post', 'technical_article', 'marketing_copy', 
                'research_summary', 'tutorial']


class TechnicalAgent(BaseAgent):
    """Agent specialized in technical content and accuracy"""
    
    def __init__(self, agent_id: int):
        super().__init__(agent_id, "Technical Agent", "technical")
        self.performance_baseline = 0.68
    
    def execute(self, task: dict, previous_content: str = "") -> dict:
        logger.debug(f"{self.name} executing technical task")
        
        topic = task.get('topic', 'Unknown')
        requirements = task.get('requirements', {})
        
        content = self._generate_technical_content(topic, requirements, previous_content)
        quality = self._calculate_quality(task, content)
        
        return {
            'content': content,
            'quality': quality,
            'agent': self.name
        }
    
    def _generate_technical_content(self, topic: str, requirements: dict, previous: str) -> str:
        if previous:
            return f"{previous}\n\n[Technical Analysis by {self.name}]\n" \
                f"Technical considerations: {self._add_technical_details()}"
        else:
            return f"[Technical Documentation by {self.name}]\n" \
                   f"Topic: {topic}\n\n" \
                   f"Technical overview: {self._add_technical_details()}"
    
    def _add_technical_details(self) -> str:
        details = [
            "Implementation requires careful consideration of system architecture and scalability.",
            "Performance metrics indicate optimal efficiency under specified constraints.",
            "Integration follows industry best practices and established protocols.",
            "The technical framework ensures robustness and maintainability."
        ]
        return random.choice(details)
    
    def get_specializations(self) -> List[str]:
        return ['technical_article', 'research_summary', 'tutorial']


class AgentTeam:
    """Manages team of specialized agents"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize specialized agents
        self.agents = [
            ResearchAgent(0),
            WritingAgent(1),
            EditorAgent(2),
            TechnicalAgent(3)
        ]
        
        # Task to agent mapping
        self.task_sequences = {
            'blog_post': [self.agents[1], self.agents[2]],  # Writing -> Editor
            'technical_article': [self.agents[0], self.agents[3], self.agents[2]],  # Research -> Technical -> Editor
            'marketing_copy': [self.agents[1], self.agents[2]],  # Writing -> Editor
            'research_summary': [self.agents[0], self.agents[3]],  # Research -> Technical
            'tutorial': [self.agents[0], self.agents[1], self.agents[2]]  # Research -> Writing -> Editor
        }
        
        logger.info(f"Agent Team initialized with {len(self.agents)} agents")
    
    def get_task_sequence(self, task_type: str) -> List[BaseAgent]:
        """Get optimal agent sequence for task type"""
        return self.task_sequences.get(task_type, self.agents[:2])
    
    def get_task_agents(self, task_type: str) -> List[BaseAgent]:
        """Get agents suitable for task type"""
        suitable = []
        for agent in self.agents:
            if task_type in agent.get_specializations():
                suitable.append(agent)
        return suitable if suitable else self.agents[:2]
    
    def get_agent_id(self, agent: BaseAgent) -> int:
        """Get agent ID"""
        return agent.agent_id
    
    def get_agent_by_id(self, agent_id: int) -> BaseAgent:
        """Get agent by ID"""
        return self.agents[agent_id]