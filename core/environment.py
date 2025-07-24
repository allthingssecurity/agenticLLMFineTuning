#!/usr/bin/env python3
"""
Core Environment Framework for Agentic RL Training

This is the foundation for creating domain-specific environments where agents
learn to take discrete actions to solve problems step-by-step.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import random


@dataclass
class ActionResult:
    """Result of taking an action in the environment."""
    observation: str
    reward: float
    done: bool
    info: Dict[str, Any]


class BaseEnvironment(ABC):
    """
    Base class for all agentic RL environments.
    
    To create a new domain:
    1. Inherit from this class
    2. Define your action space in get_valid_actions()
    3. Implement step() method for action handling
    4. Implement reset() method for problem initialization
    5. Add domain-specific reward logic
    """
    
    def __init__(self):
        self.current_state = ""
        self.problem = ""
        self.steps_taken = 0
        self.max_steps = 10
        self.trajectory = []
    
    @abstractmethod
    def reset(self, problem: str) -> str:
        """
        Reset environment with new problem.
        
        Args:
            problem: The problem to solve
            
        Returns:
            Initial observation string
        """
        pass
    
    @abstractmethod
    def step(self, action: str, action_params: Optional[str] = None) -> ActionResult:
        """
        Take an action in the environment.
        
        Args:
            action: The action to take
            action_params: Optional parameters for the action
            
        Returns:
            ActionResult with observation, reward, done flag, and info
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self) -> List[str]:
        """
        Get list of valid actions for current state.
        
        Returns:
            List of valid action strings
        """
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get current environment state."""
        return {
            "current_state": self.current_state,
            "problem": self.problem,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "valid_actions": self.get_valid_actions()
        }


class BaseAgent(ABC):
    """
    Base class for agentic RL agents.
    
    Agents learn to select optimal actions based on the current state
    and receive rewards for their choices.
    """
    
    def __init__(self):
        self.exploration_rate = 0.1  # Epsilon for exploration
    
    @abstractmethod
    def select_action(self, state: Dict[str, Any], valid_actions: List[str]) -> Tuple[str, Optional[str]]:
        """
        Select an action given the current state.
        
        Args:
            state: Current environment state
            valid_actions: List of valid actions
            
        Returns:
            Tuple of (action, action_params)
        """
        pass
    
    async def run_episode(self, env: BaseEnvironment, problem: str) -> Dict[str, Any]:
        """
        Run a complete episode in the environment.
        
        Args:
            env: The environment to run in
            problem: The problem to solve
            
        Returns:
            Dictionary with episode results and trajectory
        """
        # Reset environment
        initial_obs = env.reset(problem)
        
        trajectory = []
        total_reward = 0.0
        done = False
        
        while not done and env.steps_taken < env.max_steps:
            # Get current state and valid actions
            state = env.get_state()
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            # Select action
            action, action_params = self.select_action(state, valid_actions)
            
            # Take action
            result = env.step(action, action_params)
            
            # Record step
            trajectory.append({
                "step": env.steps_taken,
                "action": action,
                "action_params": action_params,
                "observation": state["current_state"],
                "next_observation": result.observation,
                "reward": result.reward,
                "done": result.done,
                "info": result.info
            })
            
            total_reward += result.reward
            done = result.done
        
        return {
            "problem": problem,
            "trajectory": trajectory,
            "total_reward": total_reward,
            "success": done and total_reward > 0,
            "steps_taken": env.steps_taken,
            "final_state": env.current_state
        }


class RandomAgent(BaseAgent):
    """Simple random agent for baseline testing."""
    
    def select_action(self, state: Dict[str, Any], valid_actions: List[str]) -> Tuple[str, Optional[str]]:
        """Select a random action."""
        if not valid_actions:
            return "no_action", None
        
        action = random.choice(valid_actions)
        return action, None


# Example Math Environment Implementation
class MathEnvironment(BaseEnvironment):
    """
    Example math problem-solving environment.
    Shows how to implement domain-specific logic.
    """
    
    def __init__(self):
        super().__init__()
        self.original_problem = ""
        self.solution_steps = []
    
    def reset(self, problem: str) -> str:
        """Reset with a new math problem."""
        self.problem = problem
        self.original_problem = problem
        self.current_state = f"Problem: {problem}"
        self.steps_taken = 0
        self.solution_steps = []
        self.trajectory = []
        
        return self.current_state
    
    def get_valid_actions(self) -> List[str]:
        """Get valid math actions."""
        base_actions = [
            "identify_operation",
            "set_up_equation", 
            "add_terms",
            "subtract_terms",
            "multiply_terms",
            "divide_terms",
            "state_answer"
        ]
        
        # Filter based on current state (optional)
        return base_actions
    
    def step(self, action: str, action_params: Optional[str] = None) -> ActionResult:
        """Execute a math action."""
        self.steps_taken += 1
        reward = 0.0
        done = False
        info = {"action_successful": False}
        
        if action == "identify_operation":
            if "solve" in self.problem.lower() or "x" in self.problem:
                observation = "Identified: equation solving"
                reward = 1.5
                info["action_successful"] = True
            elif any(op in self.problem for op in ['+', '-', '×', '*', '÷', '/']):
                observation = "Identified: arithmetic"
                reward = 1.5
                info["action_successful"] = True
            else:
                observation = "Operation type unclear"
                reward = 0.5
        
        elif action == "set_up_equation":
            observation = f"Equation set up: {self.problem}"
            reward = 1.0
            info["action_successful"] = True
            
        elif action == "subtract_terms":
            if "+" in self.current_state and "=" in self.current_state:
                observation = "Moved terms to isolate variable"
                reward = 2.0
                info["action_successful"] = True
            else:
                observation = "Subtraction applied"
                reward = 1.0
                
        elif action == "divide_terms":
            if "x" in self.current_state:
                observation = "Isolated variable by division"
                reward = 2.0
                info["action_successful"] = True
            else:
                observation = "Division applied"
                reward = 1.0
                
        elif action == "state_answer":
            observation = f"Final answer: {action_params or 'answer provided'}"
            reward = 3.0
            done = True
            info["action_successful"] = True
            
        else:
            observation = f"Action {action} completed"
            reward = 0.5
        
        self.current_state = observation
        self.solution_steps.append({
            "action": action,
            "params": action_params,
            "result": observation
        })
        
        return ActionResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info
        )


class SmartMathAgent(BaseAgent):
    """
    Example of a more intelligent agent with domain knowledge.
    """
    
    def select_action(self, state: Dict[str, Any], valid_actions: List[str]) -> Tuple[str, Optional[str]]:
        """Select action based on problem state."""
        current_state = state["current_state"]
        steps_taken = state["steps_taken"]
        
        # Early steps: identify the problem type
        if steps_taken == 0:
            return "identify_operation", None
        
        # If we haven't set up the equation yet
        if steps_taken == 1 and "identify" in current_state.lower():
            return "set_up_equation", None
        
        # Look for equation solving patterns
        if "=" in current_state and "x" in current_state:
            if "+" in current_state or "-" in current_state:
                return "subtract_terms", None
            elif any(char.isdigit() for char in current_state.split("x")[0]):
                return "divide_terms", None
        
        # For arithmetic problems
        if "arithmetic" in current_state.lower():
            if "×" in state["problem"] or "*" in state["problem"]:
                return "multiply_terms", None
            elif "+" in state["problem"]:
                return "add_terms", None
        
        # Final step: provide answer
        if steps_taken >= 3:
            return "state_answer", "final answer"
        
        # Default: random selection
        return random.choice(valid_actions), None