#!/bin/bash
"""
Quick Start Script for Agentic RL Framework
Run this script to get started immediately
"""

echo "🤖 Agentic RL Training Framework - Quick Start"
echo "=============================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
echo "📋 Python version: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Error: Python 3.8+ required. Please upgrade Python."
    exit 1
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv agentic_env
source agentic_env/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run math demo
echo "🎯 Running math domain demo..."
echo "This will:"
echo "  1. Generate action-based trajectories"
echo "  2. Load Qwen2.5-0.5B-Instruct with LoRA"
echo "  3. Fine-tune on math problems"
echo "  4. Test inference with learned actions"
echo ""

python demos/math_demo.py

echo ""
echo "🎉 Quick start completed!"
echo ""
echo "📚 Next steps:"
echo "  1. Review the results in math_demo_results.json"
echo "  2. Explore examples/custom_domain_template.py for your domain"
echo "  3. Read docs/domain_guide.md for detailed adaptation guide"
echo "  4. Check README.md for full documentation"
echo ""
echo "🚀 Ready to build agentic AI for your domain!"