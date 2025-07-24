#!/bin/bash

echo "🚀 Multi-Turn Agentic RL System Setup"
echo "===================================="
echo

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "✅ Python 3 found"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv multi_turn_env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source multi_turn_env/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install openai huggingface_hub transformers peft torch accelerate

echo
echo "✅ Setup complete!"
echo
echo "🔧 Next steps:"
echo "1. Set your OpenAI API key:"
echo "   export OPENAI_API_KEY='your_key_here'"
echo
echo "2. Run the examples:"
echo "   source multi_turn_env/bin/activate"
echo "   python3 api_integration_example.py"
echo
echo "3. Test individual components:"
echo "   python3 core/structured_actions.py"
echo "   python3 complete_multi_turn_demo.py"
echo
echo "📚 Read README_MULTI_TURN.md for detailed documentation"
echo
echo "🎉 Ready to build multi-turn agents!"