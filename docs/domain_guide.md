# 🎨 Domain Adaptation Guide

**Complete guide to adapting the agentic RL framework for your specific domain**

This guide walks you through every step of creating a custom agentic RL system for your domain, with practical examples and best practices.

## 🎯 Overview: From Math to Your Domain

Our framework has demonstrated success in mathematics:
- **Problem**: "Solve for x: 3x + 7 = 16"
- **Actions**: identify_operation → subtract_terms → divide_terms → state_answer
- **Result**: Model learns step-by-step equation solving

**Your domain** could be:
- 📝 Code generation and debugging
- ✍️ Creative writing and editing
- 🏢 Business strategy and decision making
- 🔬 Scientific reasoning and hypothesis testing
- 🎮 Game playing and strategy
- 📊 Data analysis and insights
- 🛠️ Task planning and execution

## 📋 Step-by-Step Adaptation Process

### Step 1: Define Your Problem Space

**Questions to Ask:**
1. What types of problems do you want to solve?
2. What does a "good solution" look like in your domain?
3. How do experts in your field approach these problems?
4. What are the key steps in the reasoning process?

**Example Domains:**

#### 📝 Code Generation
```python
# Problem types
problems = [
    "Write a function to sort a list of dictionaries by a specific key",
    "Debug this code that has a memory leak",
    "Optimize this algorithm for better performance",
    "Add error handling to this API endpoint"
]

# Success criteria
def evaluate_code_solution(code: str) -> float:
    score = 0.0
    if compiles_successfully(code): score += 3.0
    if passes_tests(code): score += 4.0
    if follows_style_guide(code): score += 1.0
    if is_efficient(code): score += 2.0
    return score
```

#### ✍️ Creative Writing
```python
# Problem types
problems = [
    "Write a compelling opening paragraph for a mystery novel",
    "Create dialogue that reveals character personality",
    "Develop a plot twist that surprises but makes sense",
    "Edit this passage to improve flow and clarity"
]

# Success criteria
def evaluate_writing_solution(text: str) -> float:
    score = 0.0
    if is_grammatically_correct(text): score += 2.0
    if has_good_flow(text): score += 2.0
    if is_engaging(text): score += 3.0
    if shows_creativity(text): score += 2.0
    if fits_genre_conventions(text): score += 1.0
    return score
```

### Step 2: Design Your Action Space

**Key Principles:**
- Actions should be **discrete and meaningful**
- Each action should produce a **clear, observable change**
- Actions should **build upon each other** logically
- Include actions for **analysis, execution, and validation**

#### Action Categories

**Analysis Actions** (understand the problem)
```python
analysis_actions = [
    "analyze_requirements",    # What needs to be done?
    "identify_constraints",    # What are the limitations?
    "assess_complexity",       # How difficult is this?
    "research_approaches"      # What methods could work?
]
```

**Planning Actions** (decide approach)
```python
planning_actions = [
    "create_outline",          # Structure the solution
    "break_into_steps",        # Decompose the problem
    "prioritize_components",   # What to do first?
    "allocate_resources"       # What do we need?
]
```

**Execution Actions** (implement solution)
```python
execution_actions = [
    "implement_core_logic",    # Main functionality
    "handle_edge_cases",       # Special scenarios
    "optimize_performance",    # Make it efficient
    "add_error_handling"       # Make it robust
]
```

**Validation Actions** (check quality)
```python
validation_actions = [
    "test_functionality",      # Does it work?
    "verify_requirements",     # Does it meet needs?
    "review_quality",          # Is it good enough?
    "finalize_solution"        # Submit final answer
]
```

### Step 3: Implement Your Environment

**Template with Detailed Comments:**

```python
class YourDomainEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__()
        # Domain-specific state tracking
        self.solution_components = []
        self.quality_metrics = {}
        self.constraints = []
    
    def reset(self, problem: str) -> str:
        """Initialize environment for new problem."""
        self.problem = problem
        self.current_state = f"New {self.domain_name} problem: {problem}"
        self.steps_taken = 0
        
        # Parse problem for domain-specific info
        self.constraints = self._extract_constraints(problem)
        self.requirements = self._extract_requirements(problem)
        
        return self.current_state
    
    def get_valid_actions(self) -> List[str]:
        """Return actions valid for current state."""
        # Base actions always available
        base_actions = ["analyze_requirements", "finalize_solution"]
        
        # State-dependent actions
        if self.steps_taken == 0:
            return ["analyze_requirements"]
        
        if "analyzed" not in self.current_state.lower():
            return ["analyze_requirements", "identify_constraints"]
        
        if not self.solution_components:
            return ["create_outline", "break_into_steps"]
        
        # Full action set when ready
        return self._get_all_domain_actions()
    
    def step(self, action: str, action_params: Optional[str] = None) -> ActionResult:
        """Execute action and return result."""
        self.steps_taken += 1
        reward = 0.0
        done = False
        info = {"action_successful": False}
        
        # Domain-specific action handling
        if action == "analyze_requirements":
            observation, reward = self._handle_analysis(action_params)
            info["action_successful"] = True
            
        elif action == "create_outline":
            observation, reward = self._handle_planning(action_params)
            info["action_successful"] = True
            
        elif action == "implement_core_logic":
            observation, reward = self._handle_implementation(action_params)
            info["action_successful"] = True
            
        elif action == "finalize_solution":
            observation, reward = self._handle_finalization(action_params)
            done = True
            info["action_successful"] = True
            
        else:
            observation = f"Unknown action: {action}"
            reward = -0.5
        
        self.current_state = observation
        return ActionResult(observation, reward, done, info)
    
    def _handle_analysis(self, params: str) -> Tuple[str, float]:
        """Handle analysis-type actions."""
        # Extract key information from problem
        key_elements = self._extract_key_elements(self.problem)
        
        observation = f"Analyzed problem: Found {len(key_elements)} key elements"
        reward = 1.5 if len(key_elements) > 0 else 0.5
        
        return observation, reward
    
    def _calculate_domain_reward(self, action: str, result: str) -> float:
        """Calculate reward based on domain-specific criteria."""
        base_reward = 1.0
        
        # Reward for progress
        if self._shows_progress(result):
            base_reward += 1.0
        
        # Reward for quality
        if self._meets_quality_standard(result):
            base_reward += 2.0
        
        # Reward for efficiency
        if self.steps_taken <= self.optimal_steps:
            base_reward += 0.5
        
        return base_reward
```

### Step 4: Create Intelligent Agents

**Smart agents use domain knowledge for better action selection:**

```python
class YourDomainAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        # Domain-specific knowledge
        self.domain_patterns = self._load_domain_patterns()
        self.success_heuristics = self._load_heuristics()
    
    def select_action(self, state: Dict[str, Any], valid_actions: List[str]) -> Tuple[str, Optional[str]]:
        """Select best action using domain expertise."""
        current_state = state["current_state"]
        steps_taken = state["steps_taken"]
        
        # Early stage: Always start with analysis
        if steps_taken == 0:
            return "analyze_requirements", None
        
        # Use domain-specific heuristics
        best_action = self._apply_domain_heuristics(state, valid_actions)
        action_params = self._generate_action_params(best_action, state)
        
        return best_action, action_params
    
    def _apply_domain_heuristics(self, state: Dict, actions: List[str]) -> str:
        """Apply domain-specific intelligence."""
        # Pattern matching
        for pattern, preferred_action in self.domain_patterns.items():
            if pattern in state["current_state"].lower():
                if preferred_action in actions:
                    return preferred_action
        
        # Fallback to random selection
        return random.choice(actions)
```

### Step 5: Generate Quality Training Data

**Diverse, Representative Problems:**

```python
def generate_domain_problems(difficulty_levels: List[str] = ["easy", "medium", "hard"]) -> List[str]:
    """Generate diverse training problems."""
    problems = []
    
    for difficulty in difficulty_levels:
        # Different problem types for each difficulty
        if difficulty == "easy":
            problems.extend([
                "Simple problem that tests basic skills",
                "Straightforward case with clear solution",
                "Basic scenario with minimal complexity"
            ])
        elif difficulty == "medium":
            problems.extend([
                "Multi-step problem requiring planning",
                "Problem with multiple valid approaches",
                "Case requiring domain-specific knowledge"
            ])
        elif difficulty == "hard":
            problems.extend([
                "Complex problem with multiple constraints",
                "Edge case requiring creative solution",
                "Ambiguous problem requiring clarification"
            ])
    
    return problems

# Quality filtering
async def generate_high_quality_trajectories():
    """Generate only successful, high-reward trajectories."""
    env = YourDomainEnvironment()
    agent = YourDomainAgent()
    
    problems = generate_domain_problems()
    high_quality_trajectories = []
    
    for problem in problems:
        episode = await agent.run_episode(env, problem)
        
        # Only keep high-quality episodes
        if episode['success'] and episode['total_reward'] > MIN_QUALITY_THRESHOLD:
            high_quality_trajectories.append(episode)
    
    return high_quality_trajectories
```

### Step 6: Domain-Specific Evaluation

**Metrics That Matter for Your Domain:**

```python
def evaluate_domain_performance(responses: List[str]) -> Dict[str, float]:
    """Evaluate model performance on domain-specific criteria."""
    metrics = {}
    
    # Generic metrics
    metrics['avg_action_count'] = sum(r.lower().count('action:') for r in responses) / len(responses)
    metrics['completion_rate'] = sum(1 for r in responses if 'final' in r.lower()) / len(responses)
    
    # Domain-specific metrics
    if DOMAIN == "code":
        metrics['syntax_correctness'] = evaluate_code_syntax(responses)
        metrics['functionality_score'] = evaluate_code_functionality(responses)
        metrics['style_adherence'] = evaluate_code_style(responses)
        
    elif DOMAIN == "writing":
        metrics['grammar_score'] = evaluate_grammar(responses)
        metrics['creativity_score'] = evaluate_creativity(responses)
        metrics['coherence_score'] = evaluate_coherence(responses)
        
    elif DOMAIN == "business":
        metrics['feasibility_score'] = evaluate_business_feasibility(responses)
        metrics['roi_consideration'] = evaluate_roi_analysis(responses)
        metrics['risk_assessment'] = evaluate_risk_analysis(responses)
    
    return metrics
```

## 🏆 Best Practices

### 1. Action Design
- **✅ Good**: "optimize_algorithm" → specific, measurable outcome
- **❌ Bad**: "make_better" → vague, hard to evaluate

### 2. Reward Engineering
- **Progressive rewards**: Small rewards for steps, big rewards for completion
- **Quality weighting**: Higher rewards for better solutions
- **Efficiency bonuses**: Extra rewards for fewer steps

### 3. Training Data Quality
- **Diversity**: Cover all problem types and difficulty levels
- **Balance**: Mix of successful and learning-from-failure trajectories  
- **Validation**: Have domain experts review training data

### 4. Evaluation Robustness
- **Multiple metrics**: Don't rely on single score
- **Human evaluation**: Include expert assessment
- **Edge case testing**: Test on unusual problems

## 🚀 Advanced Techniques

### Multi-Agent Systems
```python
class ExpertAgent(BaseAgent):
    """Specialized agent for specific subtasks."""
    def __init__(self, specialty: str):
        self.specialty = specialty  # "analysis", "implementation", "validation"
    
class OrchestratorAgent(BaseAgent):
    """Coordinates multiple expert agents."""
    def __init__(self):
        self.experts = {
            "analysis": ExpertAgent("analysis"),
            "implementation": ExpertAgent("implementation"), 
            "validation": ExpertAgent("validation")
        }
```

### Hierarchical Actions
```python
class HierarchicalEnvironment(BaseEnvironment):
    """Environment with high-level and low-level actions."""
    def get_valid_actions(self) -> List[str]:
        if self.current_level == "high":
            return ["plan_solution", "review_progress", "finalize"]
        elif self.current_level == "low":
            return ["write_code", "test_function", "debug_error"]
```

### Curriculum Learning
```python
def create_curriculum() -> List[List[str]]:
    """Progressive difficulty curriculum."""
    return [
        ["very_easy_problem_1", "very_easy_problem_2"],      # Week 1
        ["easy_problem_1", "easy_problem_2"],                # Week 2
        ["medium_problem_1", "medium_problem_2"],            # Week 3
        ["hard_problem_1", "hard_problem_2"],                # Week 4
        ["expert_problem_1", "expert_problem_2"]             # Week 5
    ]
```

## 🔧 Troubleshooting Common Issues

### Problem: Low Action Count in Responses
**Solution**: 
- Adjust system prompt to emphasize action format
- Increase action-based examples in training data
- Add action count to reward function

### Problem: Poor Domain Accuracy
**Solution**:
- Add domain experts to validation process
- Increase training data diversity
- Implement domain-specific reward functions

### Problem: Inconsistent Action Sequences
**Solution**:
- Add state dependencies to `get_valid_actions()`
- Implement action prerequisites
- Use curriculum learning for complex sequences

### Problem: Training Doesn't Converge
**Solution**:
- Reduce learning rate
- Increase training data quality
- Check reward function for proper scaling

## 📊 Success Metrics by Domain

### Code Generation
- **Compilation Rate**: 95%+ for syntactic correctness
- **Test Pass Rate**: 80%+ for functional correctness
- **Style Score**: 8/10+ for code quality
- **Efficiency Gain**: 2x+ faster than baseline

### Creative Writing
- **Grammar Score**: 9/10+ for language correctness
- **Engagement Score**: 7/10+ for reader interest
- **Creativity Score**: 6/10+ for originality
- **Coherence Score**: 8/10+ for logical flow

### Business Strategy
- **Feasibility Score**: 8/10+ for practical viability
- **ROI Analysis**: Present in 90%+ of responses
- **Risk Assessment**: Comprehensive in 85%+ of cases
- **Implementation Detail**: 7/10+ for actionability

## 🎯 Next Steps

1. **Start Small**: Begin with a narrow subset of your domain
2. **Iterate Quickly**: Test with minimal viable examples
3. **Gather Feedback**: Include domain experts in evaluation
4. **Scale Gradually**: Expand problem complexity over time
5. **Monitor Quality**: Track metrics continuously

## 💡 Domain-Specific Resources

### Code Generation
- **Datasets**: CodeSearchNet, HumanEval, MBPP
- **Evaluation**: CodeBLEU, execution accuracy
- **Tools**: AST parsing, static analysis

### Creative Writing
- **Datasets**: WritingPrompts, Story Cloze
- **Evaluation**: BLEU, ROUGE, human rating
- **Tools**: Grammar checkers, readability metrics

### Business Strategy
- **Datasets**: Harvard Business Review cases
- **Evaluation**: Expert assessment, case study analysis
- **Tools**: Business model canvas, SWOT analysis

---

**Ready to adapt the framework to your domain?**

1. Start with `examples/custom_domain_template.py`
2. Follow this guide step-by-step
3. Test with the math demo first to understand the flow
4. Iterate based on domain-specific results

🚀 **Build the future of agentic AI in your field!**