# 🤖 Multi-Agent Data Preprocessing Pipeline

A sophisticated 3-agent system for intelligent data preprocessing using Docker-based code execution.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER PROBLEM STATEMENT                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │   Agent 1: Data Strategist  │
         │  ✓ Analyzes problem         │
         │  ✓ Discovers datasets       │
         │  ✓ Creates strategy         │
         │  ✓ Researches techniques    │
         │  ✓ Delegates to Agent 2     │
         └──────────────┬──────────────┘
                        │
                        ▼
         ┌─────────────────────────────┐
         │  Agent 2: Code Generator    │
         │  ✓ Writes Python code       │
         │  ✓ Executes in Docker       │
         │  ✓ Creates features         │
         │  ✓ Requests validation      │
         └──────────────┬──────────────┘
                        │
                        ▼
         ┌─────────────────────────────┐
         │  Agent 3: Quality Assurance │
         │  ✓ Validates results        │
         │  ✓ Checks quality           │
         │  ✓ Provides feedback        │
         │  ✓ Generates reports        │
         └──────────────┬──────────────┘
                        │
                        ▼
          ┌────────────────────────────┐
          │  PREPROCESSED DATASET      │
          │  + Quality Report          │
          └────────────────────────────┘
```

## 🎯 Agent Roles

### Agent 1: Data Strategist 🎯
**Purpose**: High-level planning and strategy

**Responsibilities**:
- Analyzes problem statement and dataset structure
- Uses web search to research domain-specific techniques
- Creates comprehensive preprocessing strategy
- Delegates specific tasks to Agent 2

**Tools**:
- `load_user_context()` - Load problem statement
- `discover_dataset_structure()` - Find all CSV files
- `analyze_data_quality()` - Quick quality check
- `web_search()` - Research techniques (Tavily API)
- `delegate_to_agent_2()` - Delegate tasks

**Why it's special**:
- Doesn't execute preprocessing, only plans
- Uses AI to research best practices
- Creates domain-specific strategies

---

### Agent 2: Code Generator 💻
**Purpose**: Custom code generation and execution

**Responsibilities**:
- Writes custom Python preprocessing code
- Executes code in isolated Docker container
- Creates domain-specific features dynamically
- Handles errors and iterates

**Tools**:
- `initialize_docker_executor()` - Setup Docker
- `execute_preprocessing_code()` - Run code in Docker
- `generate_feature_engineering_code()` - Template generation
- `validate_generated_code()` - Security checks
- `get_dataset_info()` - Inspect data
- `request_agent_3_validation()` - Request QA

**Why it's special**:
- **NOT limited to predefined tools**
- Writes ANY Python code needed
- Executes in secure Docker sandbox
- Can create unlimited custom transformations

---

### Agent 3: Quality Assurance ✅
**Purpose**: Validation and quality control

**Responsibilities**:
- Validates preprocessing results
- Checks for data quality issues
- Detects potential data leakage
- Provides feedback to Agent 2
- Generates final reports

**Tools**:
- `validate_data_quality()` - Comprehensive QA
- `check_data_leakage()` - Leakage detection
- `compare_distributions()` - Before/after comparison
- `suggest_improvements()` - Optimization suggestions
- `provide_feedback_to_agent_2()` - Feedback loop
- `generate_final_report()` - Final summary

**Why it's special**:
- Acts as safety net
- Catches issues before modeling
- Provides intelligent feedback
- Creates feedback loop with Agent 2

---

## 🐳 Docker Execution

### How It Works

1. **Dockerfile** creates a Python 3.11 environment with:
   - pandas, numpy, scikit-learn, scipy
   - matplotlib, seaborn
   - Runs as non-root user for security

2. **DockerCodeExecutor** manages execution:
   - Builds Docker image once
   - Mounts dataset directory as volume
   - Executes Python code in container
   - Returns stdout/stderr/success status
   - Auto-cleans up containers

3. **Security Features**:
   - Resource limits (1GB RAM, 1 CPU)
   - Network disabled
   - Non-root user
   - Isolated filesystem
   - Auto-cleanup

### Example Code Execution

```python
# Agent 2 generates this code
code = """
import pandas as pd

df = pd.read_csv('/workspace/data/youtube.csv')
df['engagement_rate'] = df['likes'] / df['views']
df['comment_rate'] = df['comments'] / df['views']
df.to_csv('/workspace/data/youtube.csv', index=False)

print(f"Created features. Shape: {df.shape}")
"""

# Executes in Docker container
result = execute_preprocessing_code(code, csv_path)
```

---

## 📦 Installation

### Prerequisites

1. **Docker Desktop** (MUST be installed and running)
   - Download: https://www.docker.com/products/docker-desktop

2. **Python 3.11+**

3. **Required packages**:
   ```bash
   pip install google-adk docker pandas numpy scikit-learn scipy python-dotenv requests
   ```

### Environment Setup

Create `.env` file with:
```
TAVILY_API_KEY=your_tavily_key
OPENAI_API_KEY=your_openai_key
```

---

## 🚀 Usage

### Run the Pipeline

```bash
cd data_preprocessing_agent
python multi_agent_orchestrator.py
```

### What Happens

1. **Agent 1** analyzes your problem and dataset
2. **Agent 2** writes and executes custom Python code
3. **Agent 3** validates results and provides feedback
4. **Final preprocessed dataset** saved to `modified_datasets/`

### Expected Output

```
🚀 MULTI-AGENT PREPROCESSING PIPELINE
=====================================

🎯 STARTING AGENT 1: DATA STRATEGIST
[Agent 1]: Analyzing dataset structure...
[Agent 1]: Creating preprocessing strategy...
[Agent 1]: Delegating to Agent 2...

💻 STARTING AGENT 2: CODE GENERATOR
[Agent 2]: Initializing Docker...
[Agent 2]: Generating preprocessing code...
[Agent 2]: Executing code in Docker...
✅ CODE EXECUTION SUCCESSFUL!

✅ STARTING AGENT 3: QUALITY ASSURANCE
[Agent 3]: Validating data quality...
[Agent 3]: All checks passed!
[Agent 3]: Generating final report...

🎉 MULTI-AGENT PREPROCESSING PIPELINE COMPLETE!
```

---

## 🔧 Testing Components

### Test Docker Executor

```bash
python docker_executor.py
```

This will:
- Build Docker image
- Test code execution
- Verify file mounting
- Check security isolation

### Expected Docker Test Output

```
Building Docker image...
✅ Docker image built successfully!

EXECUTION RESULT:
Success: True
Output:
Hello from Docker container!
Pandas version: 2.1.4
NumPy version: 1.26.2
Files in /workspace/data: [...]
```

---

## 📁 File Structure

```
data_preprocessing_agent/
├── Dockerfile                      # Docker image definition
├── docker_executor.py              # Docker execution manager
├── agent_1_strategist.py          # Agent 1 tools
├── agent_2_code_generator.py      # Agent 2 tools
├── agent_3_qa.py                  # Agent 3 tools
├── multi_agent_orchestrator.py    # Main orchestrator
├── agent.py                       # Old single-agent (deprecated)
└── README.md                      # This file
```

---

## 🎯 Key Features

### 1. **Unlimited Flexibility**
- Not limited to predefined tools
- Agent 2 can write ANY preprocessing code
- Handles unconventional datasets

### 2. **Domain Intelligence**
- Agent 1 researches domain-specific techniques
- Creates strategies tailored to problem type
- Example: YouTube → engagement metrics, Cricket → strike rate

### 3. **Secure Execution**
- Docker isolation prevents system access
- Resource limits prevent resource exhaustion
- Network disabled for security

### 4. **Self-Correcting**
- Agent 3 validates and provides feedback
- Agent 2 can iterate if issues found
- Feedback loop ensures quality

### 5. **Intelligent Planning**
- Agent 1 thinks strategically before acting
- Uses web search for best practices
- Creates comprehensive strategies

---

## 🔍 Troubleshooting

### Docker Issues

**Problem**: "Failed to initialize Docker client"
- **Solution**: Make sure Docker Desktop is running
- Check: `docker ps` in terminal

**Problem**: "Image build failed"
- **Solution**: Check Dockerfile syntax
- Try: `docker build -t preprocessing-sandbox:latest .`

**Problem**: "Permission denied"
- **Solution**: Ensure Docker has permission to mount directories
- Windows: Check Docker Desktop settings → Resources → File Sharing

### Code Execution Issues

**Problem**: "FileNotFoundError" in Docker
- **Solution**: Ensure code uses `/workspace/data/` paths
- Example: `/workspace/data/youtube.csv` (NOT `./youtube.csv`)

**Problem**: "Timeout error"
- **Solution**: Increase timeout in `execute_preprocessing_code()`
- Default: 120 seconds

---

## 💡 Example Workflows

### YouTube Trend Analysis

**Agent 1 Strategy**:
```
- Create engagement_rate = likes / views
- Create comment_rate = comments / views
- Extract publish_hour from published_date
- Create is_weekend feature
- Drop irrelevant columns (video_id, etc.)
```

**Agent 2 Code**:
```python
df['engagement_rate'] = df['likes'] / df['views']
df['comment_rate'] = df['comments'] / df['views']
df['publish_hour'] = pd.to_datetime(df['published_date']).dt.hour
df['is_weekend'] = pd.to_datetime(df['published_date']).dt.dayofweek >= 5
df = df.drop(columns=['video_id', 'thumbnail_url'])
```

**Agent 3 Validation**:
```
✅ engagement_rate values in [0, 1] range
✅ No NaN values introduced
✅ 5 new columns created
✅ Dataset ready for modeling
```

---

## 🚧 Future Enhancements

- [ ] Add Agent 2 error retry logic with automatic fixes
- [ ] Add visualization generation (matplotlib in Docker)
- [ ] Add support for multiple file formats (Excel, JSON, Parquet)
- [ ] Add incremental preprocessing for large datasets
- [ ] Add automated feature selection
- [ ] Add data versioning with DVC integration

---

## 📝 Notes

- Always ensure Docker Desktop is running before starting
- The first run will build Docker image (takes 2-3 minutes)
- Subsequent runs use cached image (much faster)
- All preprocessing happens in isolated containers
- Original datasets are never modified directly

---

## 🤝 Integration with Pipeline

This multi-agent preprocessor integrates with:

1. **user_requirement/agent.py** → Provides problem statement
2. **data_extractor_agent/agent.py** → Downloads datasets
3. **THIS AGENT** → Preprocesses data intelligently
4. **Model training agent** (future) → Uses preprocessed data

---

## ✨ Key Advantages Over Single Agent

| Feature | Single Agent | Multi-Agent System |
|---------|-------------|-------------------|
| Flexibility | 14 predefined tools | Unlimited custom code |
| Intelligence | Rule-based | Strategic planning |
| Domain Knowledge | Generic | Domain-specific research |
| Quality Control | Basic | Comprehensive validation |
| Code Execution | Limited | Full Python in Docker |
| Error Handling | Manual | Self-correcting |
| Scalability | Fixed capabilities | Unlimited growth |

---

**Built with ❤️ using Google ADK, LiteLLM, and Docker**