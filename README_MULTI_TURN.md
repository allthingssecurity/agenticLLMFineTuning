# 🔄 Multi-Turn Agentic RL System

**Production-Ready Agent Training with Structured Actions and LLM Evaluation**

## 🎯 What This System Does

This system enables training AI agents through **realistic multi-turn conversations** where:

1. **Agents take structured actions** in parseable JSON format
2. **Conversations flow naturally** across multiple turns  
3. **Actions build on each other** with proper context
4. **LLM judges evaluate quality** beyond simple completion
5. **Training data is automatically generated** in the correct format

## 🚀 Quick Start (5 Minutes)

### 1. Installation

```bash
# Clone the repository
git clone <repo_url>
cd smart-rl-trainer/clean_project

# Create virtual environment
python3 -m venv multi_turn_env
source multi_turn_env/bin/activate

# Install dependencies
pip install openai huggingface_hub transformers peft torch accelerate
```

### 2. Set Your OpenAI Key

```bash
export OPENAI_API_KEY="your_openai_key_here"
```

### 3. Run Multi-Turn Demo

```bash
# Test structured actions
python3 core/structured_actions.py

# Run complete multi-turn system
python3 complete_multi_turn_demo.py

# Analyze detailed trajectory
python3 show_trajectory_details.py
```

## 📋 System Components

### Core Files

- **`core/structured_actions.py`** - Defines JSON action format and multi-turn trajectories
- **`core/multi_turn_environment.py`** - Environment that executes structured actions
- **`core/llm_judge.py`** - LLM evaluation system with GPT-4o integration
- **`complete_multi_turn_demo.py`** - Production-ready agent with sophisticated conversations
- **`show_trajectory_details.py`** - Detailed trajectory analysis tool

### Demo Files

- **`test_llm_judge.py`** - Test LLM judge functionality
- **`simple_e2e_demo.py`** - Simple end-to-end pipeline
- **Examples in `examples/`** - Various agent implementations

## 🔧 How to Create Your Own Multi-Turn Agent

### Step 1: Define Your Agent

```python
from complete_multi_turn_demo import ProductionReadyAgent

class MyCustomAgent(ProductionReadyAgent):
    def __init__(self):
        super().__init__("MyAgent")
        self.specialization = "Your Domain Here"
    
    async def generate_response(self, user_input: str, context: List) -> str:
        # Your custom logic here
        if "specific_task" in user_input.lower():
            action = ActionFactory.create_file(
                "output.json", 
                '{"result": "custom response"}'
            )
            
            return f"""I'll handle that for you.

```json
{action.to_json()}
```

Here's what I've done to address your request..."""
        
        return await super().generate_response(user_input, context)
```

### Step 2: Create Environment with LLM Judge

```python
from multi_turn_environment import MultiTurnEnvironment
from llm_judge import OpenAILLMJudge

# Setup LLM judge
llm_judge = OpenAILLMJudge(
    model_name="gpt-4o", 
    api_key="your_openai_key"
)

# Create environment
env = MultiTurnEnvironment(llm_judge=llm_judge)
agent = MyCustomAgent()
```

### Step 3: Generate Training Trajectories

```python
async def generate_trajectories():
    tasks = [
        "Your specific task 1",
        "Your specific task 2", 
        "Your specific task 3"
    ]
    
    successful_trajectories = []
    
    for task in tasks:
        # Start trajectory
        trajectory_id = env.start_new_trajectory(task)
        
        # Simulate conversation
        conversation = [
            "User request 1",
            "User follow-up 2", 
            "User completion signal"
        ]
        
        for user_input in conversation:
            # Generate agent response
            agent_response = await agent.generate_response(user_input)
            
            # Process turn
            turn, is_complete = await env.process_turn(user_input, agent_response)
            
            if is_complete:
                break
        
        # Finalize trajectory
        trajectory = await env.finalize_trajectory()
        if trajectory and trajectory.final_success:
            successful_trajectories.append(trajectory)
    
    return successful_trajectories
```

### Step 4: Evaluate and Train

```python
# Generate trajectories
trajectories = await generate_trajectories()

# Convert to training format
training_data = []
for trajectory in trajectories:
    training_item = trajectory.to_training_format()
    training_data.append(training_item)

# Now you have training data ready for model fine-tuning!
```

## 🎯 Example: API Integration Agent

Here's a complete example that generates the trajectory shown in the demo:

### Run the Example

```bash
python3 api_integration_example.py
```

This will generate a 4-turn conversation:

1. **User**: "I need API integration..." → **Agent**: Creates config file
2. **User**: "Can you test it?" → **Agent**: Makes API call  
3. **User**: "What about errors?" → **Agent**: Creates error handler
4. **User**: "Perfect!" → **Agent**: Completes task

### Expected Output

```
🔄 TURN 1: create_file → api_config.json (466 bytes)
🔄 TURN 2: api_call → https://api.yourservice.com/health  
🔄 TURN 3: create_file → robust_api_client.py (5,417 bytes)
🔄 TURN 4: complete_task → Success!

📊 LLM Evaluation:
   Task Completion: 9.0/10
   Action Quality: 9.5/10  
   Total Reward: 11.0
   Success: ✅
```

## 📊 Understanding the Evaluation System

### Environment Judge (Basic)
- ✅ Files created
- ✅ API calls made  
- ✅ Task marked complete
- **Limitation**: Can't evaluate content quality

### LLM Judge (Advanced)
- 📊 **Task Completion**: How well was the task accomplished?
- 🔧 **Action Quality**: Were actions appropriate and well-executed?
- ⚡ **Efficiency**: Was the solution efficient?
- 💡 **Creativity**: Was the approach creative or novel?

### Reward Structure
```python
# Small intermediate rewards (GRPO style)
intermediate_rewards = [0.5, 0.5, 0.5, 0.5]  # Per action

# Large trajectory-end reward
trajectory_end_reward = 9.0  # Based on LLM evaluation

# Total reward used for training
total_reward = sum(intermediate_rewards) + trajectory_end_reward
```

## 🔧 Customization Guide

### Adding New Action Types

```python
# In structured_actions.py
class ActionType(Enum):
    # Add your custom actions
    CUSTOM_ACTION = "custom_action"
    ANOTHER_ACTION = "another_action"

# In ActionFactory
@staticmethod
def custom_action(param1: str, param2: int) -> StructuredAction:
    return StructuredAction(
        action_type=ActionType.CUSTOM_ACTION,
        parameters={
            "param1": param1,
            "param2": param2
        }
    )
```

### Custom LLM Judge Prompts

```python
# Modify the prompt in llm_judge.py
def _build_trajectory_evaluation_prompt(self, task, trajectory, context):
    return f"""You are evaluating an AI agent on: {task}

    Custom evaluation criteria:
    - Domain-specific quality metrics
    - Your specific requirements
    - Custom scoring rubric
    
    Agent actions: {trajectory}
    
    Provide structured JSON response..."""
```

### Different Conversation Patterns

```python
# Multi-turn patterns
patterns = {
    "troubleshooting": [
        "Describe the problem",
        "What have you tried?", 
        "Let me implement a solution",
        "Test this approach"
    ],
    
    "setup_workflow": [
        "Initial requirements",
        "Configuration setup",
        "Testing and validation", 
        "Deployment ready"
    ],
    
    "iterative_improvement": [
        "Basic implementation",
        "Optimization suggestions",
        "Advanced features",
        "Production hardening"
    ]
}
```

## 🚀 Production Deployment

### Client Integration

The structured actions can be parsed and executed by any client:

```python
# Client-side action parser
def execute_agent_action(action_json: str):
    action = StructuredAction.from_json(action_json)
    
    if action.action_type == ActionType.CREATE_FILE:
        filename = action.parameters["filename"]
        content = action.parameters["content"]
        
        with open(filename, 'w') as f:
            f.write(content)
        
        return f"Created {filename}"
    
    elif action.action_type == ActionType.API_CALL:
        url = action.parameters["url"]
        method = action.parameters["method"]
        
        response = requests.request(method, url)
        return response.json()
    
    # Add more action handlers...
```

### Model Training Pipeline

```python
# Complete training pipeline
async def train_multi_turn_model():
    # 1. Generate trajectories
    trajectories = await generate_trajectories()
    
    # 2. Convert to training format  
    training_data = [t.to_training_format() for t in trajectories]
    
    # 3. Fine-tune model
    trainer = AgenticTrainer("Qwen/Qwen2.5-0.5B-Instruct")
    model, tokenizer = trainer.setup_model()
    dataset = trainer.prepare_training_data(training_data)
    await trainer.train(dataset)
    
    # 4. Save and deploy
    model.save_pretrained("./multi_turn_agent")
    # Upload to HuggingFace, etc.
```

## 🎯 Next Steps

1. **Run the demos** to understand the system
2. **Customize the agent** for your domain
3. **Generate training trajectories** with your tasks
4. **Fine-tune a model** on the multi-turn data
5. **Deploy with structured action parsing**

## 📚 Additional Resources

- **Structured Actions**: See `core/structured_actions.py` for all action types
- **LLM Evaluation**: Check `core/llm_judge.py` for custom evaluation criteria  
- **Training Integration**: Look at existing trainer integration in `core/trainer.py`
- **Production Examples**: Review `complete_multi_turn_demo.py` for sophisticated agents

## 🤝 Contributing

Feel free to:
- Add new action types
- Create domain-specific agents
- Improve LLM evaluation criteria
- Add more conversation patterns

---

**This system represents the next evolution in agent training: from simple Q&A to realistic multi-turn conversations with structured, production-ready actions.**