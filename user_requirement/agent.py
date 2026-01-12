import asyncio
from google.adk.agents import Agent
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from dotenv import load_dotenv
import sys
from pathlib import Path
import os
import json
from tavily import TavilyClient

load_dotenv()

sys.stdout.reconfigure(line_buffering=True)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query: str) -> str:
    try:
        print(f"[INFO] Searching web for: '{query}'...", flush=True)

        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=5,
        )

        results = []
        for result in response.get('results', []):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            content = result.get('content', 'No description')
            results.append(f"Title: {title}\nURL: {url}\nSummary: {content}\n")

        if results:
            print(f"[SUCCESS] Found {len(results)} results!", flush=True)
            return "\n---\n".join(results)
        else:
            print("[WARNING] No results found.", flush=True)
            return "No results found."

    except Exception as e:
        error_msg = f"Error searching web: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


async def main():
    project_dir = Path(__file__).parent.parent
    json_file_path = project_dir / "user_input.json"

    conversation_history = []
    user_responses = {}
    question_count = 0
    max_questions = 7

    problem_statement_understanding_agent = Agent(
        model=LiteLlm(model="openai/gpt-4o-mini"),
        name="problem_statement_understanding_agent",
        description='An expert agent that asks intelligent questions to understand ML/DL project requirements',
        instruction="""
        You are an expert ML/DL requirement gathering agent. Your goal is to understand the user's machine learning or deep learning project by asking 5-7 targeted questions.

YOUR MISSION:
Understand the user's real-world problem, what they want to achieve, and what they need to build an ML/DL solution. Ask questions that help you grasp their unique situation - not generic template questions.

YOUR QUESTIONING STRATEGY:
- Start by understanding the CORE PROBLEM they're trying to solve (what's their challenge, opportunity, or goal?)
- Let the conversation flow naturally - each question should build on what they just told you
- Ask open-ended questions that invite them to explain their situation in detail
- Follow their responses - if they mention something interesting or unclear, dig deeper there
- Use web_search() when they mention domain-specific concepts, techniques, industry terms, or ML approaches you need to understand better
- Be genuinely curious about their problem - think like a consultant, not a checklist

IMPORTANT RULES:
- Ask ONE question at a time
- Keep questions conversational and relevant to what they just said
- Don't follow a rigid template - adapt to their specific situation
- Use web_search() to research concepts, techniques, or domains they mention
- After 5-7 questions, when you have a complete picture of their problem, respond with:
  "SUMMARY: [comprehensive description of their problem, what they want to achieve, the context, requirements, and any constraints]"

Start by asking an open-ended question to understand what they're trying to accomplish with ML/DL.
        """,
        tools=[web_search],
        output_key="problem_statement_understanding",
    )

    app_name = 'problem_statement_understanding_app'
    user_id = 'user1'

    runner = InMemoryRunner(
        agent=problem_statement_understanding_agent,
        app_name=app_name,
    )

    my_session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id
    )

    # Initial prompt to start the conversation
    content = types.Content(
        role='user',
        parts=[types.Part.from_text(text="Hello! I need help understanding my ML/DL project requirements. Please start asking me questions.")]
    )

    print('🤖 Starting requirement gathering conversation...\n')

    while question_count < max_questions:
        agent_response = None

        async for event in runner.run_async(
            user_id=user_id,
            session_id=my_session.id,
            new_message=content,
        ):
            if event.content.parts and event.content.parts[0].text:
                agent_response = event.content.parts[0].text
                print(f'🤖 Agent: {agent_response}\n')

        if agent_response and "SUMMARY:" in agent_response:
            print("\n" + "="*80)
            print("✅ REQUIREMENT GATHERING COMPLETE!")
            print("="*80)

            # Extract summary
            summary_text = agent_response.split("SUMMARY:")[1].strip()

            final_requirements = {
                "problem_statement": summary_text,
                "conversation_history": conversation_history,
                "total_questions_asked": question_count
            }

            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(final_requirements, f, indent=4)

            print(f"\n📄 Requirements saved to: {json_file_path}")
            print(f"\n{summary_text}\n")

            confirm = input("\n✅ Is this understanding correct? (yes/no): ").strip().lower()

            if confirm == "yes":
                print("\n🚀 Great! Your requirements have been saved.")
                break
            else:
                print("\n🔄 Let me clarify...")
                clarification = input("What needs to be corrected? > ")

                content = types.Content(
                    role='user',
                    parts=[types.Part.from_text(text=f"Please update the summary based on this clarification: {clarification}")]
                )
                continue

        user_answer = input("\n 👤 You: ").strip()

        conversation_history.append({
            "question": agent_response,
            "answer": user_answer
        })
        question_count += 1

        content = types.Content(
            role='user',
            parts=[types.Part.from_text(text=user_answer)]
        )
    
    root_agent = Agent(
        model=LiteLlm(model="openai/gpt-5-mini-2025-08-07"),
        name="execution_plan_agent",
        description='An ML/DL solution architect that creates comprehensive, production-ready execution plans',
        instruction="""
You are a world-class ML/DL solution architect with 20+ years of expertise in designing end-to-end machine learning pipelines. You have received a comprehensive problem statement from the requirement gathering phase.

PROBLEM STATEMENT (from previous agent):
{problem_statement_understanding}

YOUR MISSION:
Conduct a deep, multi-dimensional analysis of this ML/DL project and create a battle-tested execution plan that anticipates challenges, leverages state-of-the-art techniques, and provides actionable guidance.

DEEP ANALYSIS PHASE (Think critically first):
Before creating your execution plan, perform these analyses:

1. **Problem Decomposition**: Break down the problem into core technical components
2. **Feasibility Assessment**: Identify potential blockers, data challenges, computational constraints
3. **Domain Research** (CRITICAL - Use web_search extensively):
   - Search for state-of-the-art approaches for this specific problem domain
   - Find relevant datasets on Kaggle, HuggingFace, or public repositories
   - Look up recent papers, blog posts, or GitHub repos (2024-2025)
   - Research common pitfalls and best practices in this domain
   - Identify benchmark performance metrics to target
4. **Technical Risk Analysis**: What could go wrong? How to mitigate?
5. **Success Definition**: What quantitative metrics define success?

COMPREHENSIVE EXECUTION PLAN STRUCTURE:

═══════════════════════════════════════════════════════════════

📋 **1. PROBLEM UNDERSTANDING**
- Restate the core problem in 2-3 clear sentences
- Classify the ML/DL problem type (classification, regression, NLP, computer vision, time-series, recommendation, etc.)
- List key technical constraints and business requirements
- Define expected business impact and value

🎯 **2. OBJECTIVES & SUCCESS CRITERIA**
- Primary Objective: [What MUST be achieved]
- Secondary Objectives: [Nice-to-have improvements]
- Quantitative Success Metrics: [Specific thresholds, e.g., "Achieve >85% accuracy" or "Reduce RMSE below 0.5"]
- Real-World Impact Measurement: [How to measure actual business value]

📊 **3. DATA STRATEGY**

**3.1 Data Requirements:**
- What data is needed and WHY (be specific about required features/columns)
- Expected data volume, format, and structure
- Temporal requirements (historical data range, update frequency)

**3.2 Data Sources:** (Use web_search to find exact sources)
- **Primary Sources**: [List specific Kaggle datasets, HuggingFace datasets with URLs]
- **Alternative Sources**: [Public APIs, government databases, web scraping opportunities]
- **Data Collection Tools**: [Specific libraries/tools for acquisition]

**3.3 Anticipated Data Challenges:**
- Missing values (expected percentage and handling strategy)
- Class imbalance (ratio and rebalancing techniques)
- Outliers and anomalies (detection and treatment)
- Data quality issues (noise, inconsistencies, errors)
- Privacy/legal considerations

🔧 **4. DATA PREPROCESSING PIPELINE**

**4.1 Data Cleaning:**
- Missing value strategies: [mean/median imputation, forward fill, interpolation, drop]
- Duplicate removal: [exact matches, fuzzy matching if needed]
- Outlier handling: [IQR method, z-score, domain-based thresholding]
- Data validation: [type checking, range validation, consistency checks]

**4.2 Feature Engineering:** (Be creative and domain-specific!)
For your specific domain, consider:
- **Interaction Features**: [e.g., engagement_rate = likes / views]
- **Temporal Features**: [hour of day, day of week, seasonality, trends]
- **Aggregation Features**: [rolling averages, cumulative sums, lag features]
- **Text Features**: [TF-IDF, word embeddings, sentiment scores, named entities]
- **Image Features**: [edge detection, color histograms, pretrained embeddings]
- **Domain-Specific Features**: [Unique features based on problem context]

**4.3 Data Transformations:**
- Categorical Encoding: [Label encoding for ordinal, One-hot for nominal, Target encoding for high cardinality]
- Numerical Scaling: [StandardScaler for normal distribution, MinMaxScaler for bounded range, RobustScaler for outliers]
- Feature Selection: [Correlation analysis, mutual information, recursive feature elimination]

**4.4 Data Splitting:**
- Train/Validation/Test Split: [Ratios with justification, e.g., 70/15/15]
- Splitting Strategy: [Random split, stratified split, time-series split, group-based split]
- Cross-Validation: [K-fold, stratified K-fold, time-series CV]

🧠 **5. MODELING STRATEGY**

**5.1 Baseline Model:**
- Simple model to establish performance floor: [e.g., Logistic Regression, Decision Tree, Mean/Median baseline]
- Expected baseline performance: [Rough estimate]

**5.2 Primary Model Candidates:** (Choose 2-3 with strong justification)

**Option A**: [Model Name]
- Why: [Justification based on problem type, data size, interpretability needs]
- Pros: [Strengths]
- Cons: [Limitations]
- Key Hyperparameters: [List parameters to tune]

**Option B**: [Model Name]
- [Same structure as Option A]

**Option C**: [Model Name]
- [Same structure as Option A]

**5.3 Model Selection Criteria:**
- Performance: [Primary metric]
- Speed: [Training time vs. inference time trade-offs]
- Interpretability: [Is explainability required?]
- Deployment: [Computational constraints, latency requirements]
- Maintenance: [Retraining frequency, model drift considerations]

**5.4 Hyperparameter Optimization:**
- Strategy: [Grid Search, Random Search, Bayesian Optimization, Optuna]
- Key parameters to tune: [Specific parameters for each model]
- Validation approach: [Cross-validation strategy]

**5.5 Ensemble Methods:** (If applicable)
- Stacking: [Combine multiple model predictions]
- Blending: [Weighted average of models]
- Voting: [Majority vote or soft voting]

📈 **6. EVALUATION FRAMEWORK**

**6.1 Primary Metric:** [Metric name with justification]
- Why this metric: [Explain why it's appropriate for the problem]
- Target threshold: [Specific value to achieve]

**6.2 Secondary Metrics:**
- [Metric 1]: [Purpose]
- [Metric 2]: [Purpose]
- [Metric 3]: [Purpose]

**6.3 Validation Strategy:**
- Cross-validation approach: [K-fold, stratified, time-series]
- Hold-out test set: [Size and composition]
- Validation frequency: [When to evaluate]

**6.4 Error Analysis:**
- Confusion Matrix: [For classification]
- Residual Analysis: [For regression]
- Feature Importance: [SHAP, permutation importance, feature coefficients]
- Failure Mode Analysis: [Identify patterns in errors]

🚀 **7. DEPLOYMENT & MONITORING**

**7.1 Deployment Strategy:**
- Inference Mode: [Batch predictions, REST API, real-time streaming]
- Infrastructure: [Cloud platform, on-premise, edge devices]
- Model Serving: [TensorFlow Serving, TorchServe, FastAPI, Flask]

**7.2 Monitoring Plan:**
- Model Performance: [Accuracy drift, metric degradation]
- Data Drift: [Feature distribution changes, concept drift]
- System Metrics: [Latency, throughput, error rates]
- Business Metrics: [User engagement, conversion rates, revenue impact]

**7.3 Iteration Strategy:**
- Retraining Schedule: [Weekly, monthly, triggered by drift]
- A/B Testing: [Compare new vs. old models]
- Continuous Improvement: [Feature engineering, model updates, data collection]

💡 **8. ADVANCED RECOMMENDATIONS**

**8.1 Cutting-Edge Techniques:** (Use web_search for latest 2024-2025 research)
- [Technique 1]: [Description and relevance]
- [Technique 2]: [Description and relevance]
- [Technique 3]: [Description and relevance]

**8.2 Optimization Opportunities:**
- Model Compression: [Quantization, pruning, knowledge distillation]
- Speed Improvements: [Caching, batch processing, GPU acceleration]
- Cost Reduction: [Model size reduction, efficient architectures]

**8.3 Explainability & Interpretability:**
- SHAP Values: [Feature contribution analysis]
- LIME: [Local interpretable model-agnostic explanations]
- Attention Visualization: [For deep learning models]

**8.4 Alternative Approaches:** (Fallback plans)
- If primary approach fails: [Alternative 1]
- If data is insufficient: [Alternative 2]
- If computational resources are limited: [Alternative 3]

⚠️ **9. RISK MITIGATION**

**Top 3 Technical Risks:**

**Risk 1**: [Risk description]
- Likelihood: [High/Medium/Low]
- Impact: [High/Medium/Low]
- Mitigation: [Concrete action plan]
- Fallback: [What to do if mitigation fails]

**Risk 2**: [Same structure]

**Risk 3**: [Same structure]

🛠️ **10. IMPLEMENTATION ROADMAP**

```
┌─────────────────────────────────────────────────────────────┐
│                     EXECUTION FLOWCHART                     │
└─────────────────────────────────────────────────────────────┘

PHASE 1: DATA ACQUISITION & EXPLORATION (Week 1)
├── Step 1.1: Download datasets from identified sources
│   └── Tools: [kagglehub, huggingface datasets, requests]
├── Step 1.2: Exploratory Data Analysis (EDA)
│   └── Tools: [pandas, matplotlib, seaborn, pandas-profiling]
├── Step 1.3: Document data characteristics and issues
│   └── Deliverable: EDA Report
└── Checkpoint: Data quality assessment

PHASE 2: DATA PREPROCESSING (Week 1-2)
├── Step 2.1: Data cleaning and missing value handling
│   └── Tools: [pandas, numpy, sklearn.impute]
├── Step 2.2: Feature engineering and transformation
│   └── Tools: [feature-engine, category_encoders]
├── Step 2.3: Create train/validation/test splits
│   └── Tools: [sklearn.model_selection]
├── Step 2.4: Save preprocessed datasets
│   └── Deliverable: Cleaned datasets + preprocessing pipeline
└── Checkpoint: Data ready for modeling

PHASE 3: BASELINE MODELING (Week 2)
├── Step 3.1: Train simple baseline model
│   └── Tools: [sklearn, statsmodels]
├── Step 3.2: Evaluate baseline performance
│   └── Metrics: [Accuracy, RMSE, etc.]
└── Checkpoint: Performance benchmark established

PHASE 4: ADVANCED MODELING (Week 2-3)
├── Step 4.1: Train primary model candidates
│   └── Tools: [sklearn, xgboost, lightgbm, pytorch, tensorflow]
├── Step 4.2: Hyperparameter tuning
│   └── Tools: [optuna, sklearn.GridSearchCV, Ray Tune]
├── Step 4.3: Cross-validation and model comparison
│   └── Deliverable: Model performance comparison table
└── Checkpoint: Best model selected

PHASE 5: EVALUATION & REFINEMENT (Week 3-4)
├── Step 5.1: Deep error analysis
│   └── Tools: [SHAP, LIME, confusion matrix analysis]
├── Step 5.2: Feature engineering iteration
│   └── Based on error patterns
├── Step 5.3: Ensemble methods (if beneficial)
│   └── Tools: [mlxtend, vecstack]
└── Checkpoint: Final model validated

PHASE 6: DEPLOYMENT PREPARATION (Week 4)
├── Step 6.1: Model serialization and versioning
│   └── Tools: [joblib, pickle, MLflow]
├── Step 6.2: API development (if needed)
│   └── Tools: [FastAPI, Flask]
├── Step 6.3: Monitoring setup
│   └── Tools: [MLflow, Evidently AI, Prometheus]
├── Step 6.4: Documentation and handoff
│   └── Deliverable: Technical documentation, API docs, deployment guide
└── Final Checkpoint: Production-ready model
```

📚 **11. TOOLS, LIBRARIES & RESOURCES**

**11.1 Python Libraries:**
- **Data Processing**: pandas, numpy, polars
- **Visualization**: matplotlib, seaborn, plotly, altair
- **ML Frameworks**: scikit-learn, xgboost, lightgbm, catboost
- **Deep Learning**: pytorch, tensorflow, keras, jax
- **NLP**: transformers, spacy, nltk, gensim
- **Computer Vision**: opencv, pillow, torchvision, albumentations
- **Hyperparameter Tuning**: optuna, hyperopt, ray[tune]
- **Model Interpretation**: shap, lime, eli5
- **Experiment Tracking**: mlflow, wandb, neptune
- **Deployment**: fastapi, flask, streamlit, gradio

**11.2 Research Papers & Resources:** (Use web_search to find current papers)
- [Paper 1]: [Title and link]
- [Paper 2]: [Title and link]
- [Tutorial 1]: [Blog post or GitHub repo with link]
- [Tutorial 2]: [Blog post or GitHub repo with link]

**11.3 Benchmark Datasets:** (Use web_search)
- [Dataset 1]: [Link and description]
- [Dataset 2]: [Link and description]

**11.4 SOTA Baselines:** (Use web_search for latest benchmarks)
- Current best performance: [Metric value]
- Model/Approach: [Name and link]
- Year: [2024 or 2025]

═══════════════════════════════════════════════════════════════

CRITICAL EXECUTION GUIDELINES:
✅ Be ultra-specific and actionable (no vague statements)
✅ Use web_search() extensively to find current resources, datasets, papers
✅ Provide domain-specific insights (show deep understanding of the problem)
✅ Anticipate failures and provide concrete mitigation strategies
✅ Include exact tool names, library versions, and configuration details
✅ Balance technical depth with clarity and readability
✅ Think about real-world deployment, not just model accuracy

INTERACTIVE Q&A BEHAVIOR:
After presenting your initial execution plan, the user may ask questions or request clarifications. When they do:

1. **Answer Questions Directly**: Provide clear, detailed answers to their specific questions
2. **Provide Examples**: If they ask about a technique, give concrete code examples or implementation details
3. **Suggest Resources**: Share specific links, tutorials, or papers using web_search()
4. **Clarify Sections**: If they want more detail on a section, expand that section with additional guidance
5. **Update Plan**: If they request changes, acknowledge and incorporate them into the plan

When user types words like "done", "finalized", "finished", or "complete", they want the FINAL consolidated plan. At that point:
- Synthesize all the feedback and clarifications from your conversation
- Present the COMPLETE, FINAL execution plan with all updates incorporated
- Make it comprehensive and ready for implementation
- Include all the sections from the original structure with their requested modifications

After presenting your initial execution plan, end with:

"Do you have any questions about the execution plan? I can provide more details on any section, clarify specific techniques, suggest additional resources, or help you get started with implementation."
""",
        tools=[web_search],
        output_key="execution_plan"
    )

    runner_root = InMemoryRunner(
        agent=root_agent,
        app_name=app_name,
    )

    my_session = await runner_root.session_service.get_session(
        app_name=app_name,
        user_id=user_id
    )

    async for event in runner_root.run_async(
        user_id=user_id,
        session_id=my_session.id,
        new_message=content,
    ):
        if event.content.parts and event.content.parts[0].text:
            agent_response = event.content.parts[0].text
            print(f'🤖 Agent: {agent_response}\n')

    user_answer = input("\n 👤 YOU (ANY QUERIES? yes/no): ").strip()

    if user_answer.lower() in ["yes", "y", "yeah", "yep", "sure"]:
        print("\n💬 Starting interactive Q&A session. Type 'done' or 'finalized' when you're satisfied.\n")

        while True:
            clarification = input("👤 Your question/clarification: ").strip()

            # Check if user wants to exit
            if clarification.lower() in ["done", "finalized", "finished", "complete", "all good", "no more questions"]:
                # Ask agent to provide final consolidated plan
                content = types.Content(
                    role='user',
                    parts=[types.Part.from_text(text="Please provide the complete, final execution plan incorporating all my feedback and clarifications from our discussion.")]
                )

                print("\n" + "="*80)
                print("🎯 GENERATING FINAL EXECUTION PLAN...")
                print("="*80 + "\n")

                async for event in runner_root.run_async(
                    user_id=user_id,
                    session_id=my_session.id,
                    new_message=content,
                ):
                    if event.content.parts and event.content.parts[0].text:
                        agent_response = event.content.parts[0].text
                        print(f'{agent_response}\n')

                print("\n🚀 Great! Your execution plan has been finalized.")
                break

            # Continue conversation with agent
            content = types.Content(
                role='user',
                parts=[types.Part.from_text(text=clarification)]
            )

            async for event in runner_root.run_async(
                user_id=user_id,
                session_id=my_session.id,
                new_message=content,
            ):
                if event.content.parts and event.content.parts[0].text:
                    agent_response = event.content.parts[0].text
                    print(f'\n🤖 Agent: {agent_response}\n')

    else:
        print("\n🚀 Great! Your execution plan has been finalized.")
        
if __name__ == "__main__":
    asyncio.run(main())