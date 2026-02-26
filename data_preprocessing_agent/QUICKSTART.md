# 🚀 Quick Start Guide

## Prerequisites (5 minutes)

### 1. Install Docker Desktop
- Download from: https://www.docker.com/products/docker-desktop
- Install and **START** Docker Desktop
- Verify: Open terminal and run `docker ps`

### 2. Install Python Packages
```bash
pip install google-adk docker pandas numpy scikit-learn scipy python-dotenv requests
```

### 3. Setup Environment Variables
Create `.env` file in project root:
```
TAVILY_API_KEY=your_tavily_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Pipeline (2 minutes)

### Step 1: Make sure you have data
```bash
# Check that user_input.json exists with problem statement and dataset
cat ../user_input.json
```

### Step 2: Run the orchestrator
```bash
cd data_preprocessing_agent
python multi_agent_orchestrator.py
```

### Step 3: Watch the magic happen ✨
```
🚀 MULTI-AGENT PREPROCESSING PIPELINE
=====================================

🎯 Agent 1: Analyzing dataset...
💻 Agent 2: Generating and executing code...
✅ Agent 3: Validating results...

🎉 PIPELINE COMPLETE!
```

## What Just Happened?

1. **Agent 1** analyzed your problem and created a strategy
2. **Agent 2** wrote custom Python code and executed it in Docker
3. **Agent 3** validated the results and generated a report

## Output Location

Preprocessed datasets are saved to:
```
../modified_datasets/
├── dataset_name_preprocessed.csv
├── dataset_name_preprocessed_metadata.json
```

## Testing Individual Components

### Test Docker Executor
```bash
python docker_executor.py
```

Expected output:
```
Building Docker image...
✅ Docker image built successfully!
Hello from Docker container!
```

### Test Agent 1 Tools
```bash
python -c "from agent_1_strategist import load_user_context; print(load_user_context())"
```

### Test Agent 2 Tools
```bash
python -c "from agent_2_code_generator import initialize_docker_executor; print(initialize_docker_executor())"
```

### Test Agent 3 Tools
```bash
python -c "from agent_3_qa import validate_data_quality; print(validate_data_quality('path/to/csv.csv'))"
```

## Common Issues

### "Docker not running"
**Fix**: Open Docker Desktop application

### "Image not found"
**Fix**: Run `python docker_executor.py` to build image

### "Permission denied"
**Fix**:
- Windows: Docker Desktop → Settings → Resources → File Sharing
- Mac/Linux: `sudo usermod -aG docker $USER` then logout/login

### "Module not found"
**Fix**: Make sure you're in the correct directory
```bash
cd data_preprocessing_agent
python multi_agent_orchestrator.py
```

## Next Steps

After preprocessing is complete:
1. Check `../modified_datasets/` for processed data
2. Review the quality report from Agent 3
3. Use preprocessed data for model training

## Need Help?

Check the full README.md for detailed documentation.