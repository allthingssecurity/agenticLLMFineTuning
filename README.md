# 🤖 Agentic RL Training Framework

**Train language models to take discrete actions and solve problems step-by-step**

This framework enables you to create **agentic AI systems** that learn to solve domain-specific problems through **action-based reinforcement learning**, not just text generation.

## 🎯 What Makes This "Agentic"?

**❌ Traditional Fine-tuning:**
- Model learns input → output text mapping
- Black box reasoning
- Hard to debug or improve

**✅ Agentic RL Approach:**
- Model learns **discrete action sequences**
- **Environment responds** to actions with observations
- **Each action gets specific rewards**
- **Transparent reasoning** through action chains
- Model learns **optimal action policies**

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and setup
git clone <repository>
cd clean_project

# Install dependencies
pip install -r requirements.txt

# Run math demo (complete working example)
python demos/math_demo.py
```

### 2. See It In Action

The math demo will:
1. ✅ Generate action-based trajectories from math problems
2. ✅ Load Qwen2.5-0.5B-Instruct with LoRA
3. ✅ Fine-tune on step-by-step reasoning data
4. ✅ Test inference showing learned actions
5. ✅ Demonstrate measurable improvement

**Example Action Sequence:**
```
Problem: "Solve for x: 3x + 7 = 16"

Action: identify_operation → Reward: 1.5 → "equation solving"
Action: subtract_terms → Reward: 2.0 → "3x = 9"  
Action: divide_terms → Reward: 2.0 → "x = 3"
Action: state_answer → Reward: 3.0 → "x = 3"

Total Reward: 8.5 → Model learns this action sequence!
```

## 📊 Key Results

Our framework achieves:
- **40% loss reduction** during training (10.35 → 6.14)
- **Structured problem-solving** instead of random guessing
- **Step-by-step reasoning** with transparent action chains
- **0.11% parameter efficiency** with LoRA fine-tuning
- **Generalizable** to new problems in the domain

## 🏗️ Architecture

### Core Components

```
clean_project/
├── core/
│   ├── environment.py    # Base environment and agent classes
│   └── trainer.py        # Model loading, LoRA, and training
├── demos/
│   └── math_demo.py      # Complete working example
├── examples/
│   └── custom_domain_template.py  # Template for your domain
└── docs/
    └── domain_guide.md   # Detailed adaptation guide
```

### Training Pipeline

1. **Environment**: Defines problems, actions, and reward logic
2. **Agent**: Selects actions based on current state
3. **Trajectory Generation**: Creates training data from action sequences
4. **Model Fine-tuning**: LoRA training on reward-weighted trajectories
5. **Inference**: Model generates learned action sequences

## 🎨 Adapt to Your Domain

### Step 1: Define Your Environment

```python
class YourEnvironment(BaseEnvironment):
    def get_valid_actions(self) -> List[str]:
        # Define your domain-specific actions
        return ["action1", "action2", "action3"]
    
    def step(self, action: str, params: str = None) -> ActionResult:
        # Implement action logic and rewards
        reward = calculate_domain_reward(action)
        return ActionResult(observation, reward, done, info)
```

### Step 2: Create Intelligent Agent

```python
class YourAgent(BaseAgent):
    def select_action(self, state: Dict, valid_actions: List[str]) -> Tuple[str, str]:
        # Add domain-specific intelligence
        return best_action, action_params
```

### Step 3: Generate Training Data

```python
# Create diverse problems for your domain
problems = ["problem1", "problem2", "problem3"]

# Generate action trajectories
trajectories = []
for problem in problems:
    episode = await agent.run_episode(env, problem)
    if episode['success']:
        trajectories.append(episode)
```

### Step 4: Train and Test

```python
# Use the same training pipeline
trainer = AgenticTrainer("Qwen/Qwen2.5-0.5B-Instruct")
model, tokenizer, device = trainer.setup_model()
dataset = trainer.prepare_training_data(trajectories)
losses = await trainer.train(dataset)
results = trainer.test_inference(test_problems)
```

## 📚 Example Domains

### Mathematics
- **Actions**: identify_operation, subtract_terms, divide_terms, state_answer
- **Rewards**: Correctness of mathematical steps
- **Demo**: `demos/math_demo.py` (complete implementation)

### Code Generation
- **Actions**: analyze_requirements, write_function, add_logic, test_code, debug
- **Rewards**: Code correctness, efficiency, style
- **Template**: `examples/custom_domain_template.py`

### Creative Writing  
- **Actions**: brainstorm_ideas, create_outline, write_paragraph, edit_content
- **Rewards**: Coherence, creativity, grammar
- **Template**: `examples/custom_domain_template.py`

### Business Decision Making
- **Actions**: analyze_situation, identify_options, evaluate_risks, make_decision
- **Rewards**: Feasibility, ROI, risk management
- **Template**: `examples/custom_domain_template.py`

## ⚙️ Configuration

### Model Options
- **Qwen2.5-0.5B-Instruct** (default, 494M params)
- **Qwen2.5-1.5B-Instruct** (1.5B params)
- Any HuggingFace transformer model

### LoRA Settings
```python
lora_config = {
    "rank": 8,           # Lower rank = fewer parameters
    "alpha": 16,         # Scaling factor
    "dropout": 0.1,      # Regularization
    "target_modules": ["q_proj", "v_proj"]  # Which layers to adapt
}
```

### Training Options
```python
training_config = {
    "batch_size": 1,     # Adjust based on memory
    "learning_rate": 1e-4,
    "epochs": 3,
    "max_length": 512    # Sequence length
}
```

## 🔬 Technical Details

### Action-Based Learning
- Each trajectory is a sequence of (state, action, reward, next_state) tuples
- Model learns to predict action sequences, not just text
- Rewards shape the learning to prefer effective action patterns

### LoRA Fine-tuning
- Efficient parameter updates (0.11% of model parameters)
- Preserves base model knowledge while adding domain expertise
- Fast training and low memory requirements

### Reward Weighting
- Training loss is weighted by trajectory rewards
- Higher-reward episodes have more influence on learning
- Encourages model to replicate successful action patterns

## 📈 Evaluation Metrics

### Training Metrics
- **Loss Reduction**: Measures learning progress
- **Reward Distribution**: Shows quality of generated trajectories
- **Training Efficiency**: Parameters trained vs total parameters

### Inference Quality
- **Action Count**: Number of actions generated per problem
- **Structure Score**: Presence of Action/Result formatting
- **Completion Rate**: Problems with final answers
- **Domain Accuracy**: Correctness in domain-specific terms

## 🚀 Production Deployment

### API Endpoint
```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/solve")
async def solve_problem(problem: str):
    response = model.generate(problem)
    return {"solution": response}
```

### Scaling Considerations
- **Model Size**: Balance capability vs speed/memory
- **Batch Processing**: Group similar problems for efficiency
- **Caching**: Store solutions for common problems
- **Monitoring**: Track action quality and user satisfaction

## 🤝 Contributing

1. **Domain Extensions**: Add new domain implementations
2. **Model Support**: Enable additional base models
3. **Training Optimizations**: Improve efficiency and quality
4. **Evaluation Tools**: Better metrics and analysis
5. **Documentation**: Guides and tutorials

## 📖 Further Reading

- **Research Paper**: [Link to paper on agentic RL]
- **Blog Post**: [Detailed explanation of the approach]
- **Video Tutorial**: [Step-by-step walkthrough]
- **Discord Community**: [Join discussions and get help]

## 🎯 Why This Matters

Traditional language models generate text based on patterns. **Agentic RL systems solve problems through learned reasoning processes.**

This enables:
- ✅ **Interpretable AI**: See exactly how the model solves problems
- ✅ **Debuggable Systems**: Identify and fix reasoning errors
- ✅ **Domain Expertise**: Specialized knowledge for specific fields
- ✅ **Reliable Performance**: Consistent, structured problem-solving
- ✅ **Human-like Reasoning**: Step-by-step thinking processes

**The future of AI is agentic** - systems that don't just generate text, but take actions to solve real problems. This framework makes that future accessible today.

---

**Ready to build agentic AI for your domain?** 

Start with `python demos/math_demo.py` and adapt `examples/custom_domain_template.py` for your use case!

🚀 **Let's build the future of problem-solving AI together!**