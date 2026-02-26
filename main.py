from openai import OpenAI
from pathlib import Path
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

user_query = input("Please tell me what do you want to do: ")

conversation_history = [
    {
        'role': 'user',
        'content': [{"type": "input_text", "text": user_query}]
    }
]

for i in range(1, 8):
    response = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        instructions="""You are an expert Machine Learning, Deep Learning and Data Science assistant with over 20+ years of experience.
        You have a lot of publications, did some of the great research works in the world and delivered seminars in top universities. You are also a great teacher and mentor, and you have helped many students and professionals to achieve their goals in the field of AI.
        You main task is to understand the user's query by asking relevant questions.
        
        Your job is to ask exactly 7 questions ONE AT A TIME to better understand their requirements, preferences, and constraints related to their goal.
        Ask only one question per response. Be concise and relevant.""",
        input=conversation_history
    )

    question = response.output_text
    print(f"\nQuestion {i}/7: {question}")

    user_answer = input("You: ")

    conversation_history.append({
        'role': 'assistant',
        'content': [{"type": "output_text", "text": question}]
    })
    conversation_history.append({
        'role': 'user',
        'content': [{"type": "input_text", "text": user_answer}]
    })

conversation_history.append({
    'role': 'user',
    'content': [{"type": "input_text", "text": "Based on all my answers, generate a comprehensive summary report with action plan and write it to a text file."}]
})

tools = [
    {"type": "web_search"},
    {
        "type": "function",
        "name": "write_text_file",
        "description": "Writes the given content/report into a local text file. Use this to save the comprehensive summary report.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The name of the file to write to (e.g., 'report.txt')"
                },
                "content": {
                    "type": "string",
                    "description": "The full content/report to write into the file"
                }
            },
            "required": ["filename", "content"]
        }
    }
]

def write_text_file(filename: str, content: str):
    filepath = Path(__file__).parent / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n✅ Report successfully saved to: {filepath}")
    return f"File '{filepath}' written successfully."

print("\nGenerating final report, please wait...\n")

final_response = client.responses.create(
    model="gpt-4o",
    instructions="""You are an expert Machine Learning, Deep Learning and Data Science assistant with over 20+ years of experience.
    You have published groundbreaking research, delivered keynotes at NeurIPS, ICML, and CVPR, and mentored hundreds of students and professionals worldwide.

    Your task is to generate an EXTREMELY DETAILED and COMPREHENSIVE summary report based on the user's answers.
    
    The report MUST follow this exact structure with deep analysis in each section:

    ═══════════════════════════════════════════════════════
    📌 SECTION 1: USER'S GOAL & PROBLEM STATEMENT
    ═══════════════════════════════════════════════════════
    - Restate the user's goal clearly and precisely
    - Break down the core problem they are trying to solve
    - Identify the expected outcome and success criteria

    ═══════════════════════════════════════════════════════
    🔬 SECTION 2: STATE OF THE ART ANALYSIS
    ═══════════════════════════════════════════════════════
    - Current leading approaches and techniques in this domain
    - Recent breakthroughs and research papers worth knowing
    - Benchmark datasets commonly used in this field
    - Current performance benchmarks and SOTA metrics
    - Key research labs and companies working in this area

    ═══════════════════════════════════════════════════════
    📊 SECTION 3: ANALYSIS OF USER'S REQUIREMENTS & CONSTRAINTS
    ═══════════════════════════════════════════════════════
    - Detailed breakdown of all requirements mentioned
    - Analysis of constraints (time, budget, compute, data, skills)
    - Feasibility assessment given the constraints
    - Risk level: Low / Medium / High with justification

    ═══════════════════════════════════════════════════════
    🗂️ SECTION 4: DATASET & DATA STRATEGY
    ═══════════════════════════════════════════════════════
    - Recommended publicly available datasets with links
    - Data collection strategies if datasets are unavailable
    - Data preprocessing and augmentation techniques
    - Data splitting strategy (train/val/test)
    - Potential data quality issues and how to handle them

    ═══════════════════════════════════════════════════════
    🛠️ SECTION 5: RECOMMENDED TOOLS, FRAMEWORKS & TECHNIQUES
    ═══════════════════════════════════════════════════════
    - Programming languages and libraries (with versions)
    - Deep learning frameworks and why they are recommended
    - Pre-trained models and transfer learning options
    - Experiment tracking tools (MLflow, W&B, etc.)
    - Hardware recommendations

    ═══════════════════════════════════════════════════════
    🗺️ SECTION 6: STEP-BY-STEP ACTION PLAN
    ═══════════════════════════════════════════════════════
    Provide a detailed week-by-week or phase-by-phase plan:
    - Phase 1: Setup & Research (with specific tasks)
    - Phase 2: Data Collection & Preprocessing (with specific tasks)
    - Phase 3: Model Development & Training (with specific tasks)
    - Phase 4: Evaluation & Optimization (with specific tasks)
    - Phase 5: Deployment / Presentation (with specific tasks)
    Include estimated time for each phase.

    ═══════════════════════════════════════════════════════
    ⚠️ SECTION 7: CHALLENGES, RISKS & MITIGATION STRATEGIES
    ═══════════════════════════════════════════════════════
    - List every possible challenge with detailed explanation
    - Technical risks and how to mitigate each one
    - Non-technical risks (time, scope creep, etc.)
    - Fallback strategies if primary approach fails

    ═══════════════════════════════════════════════════════
    📚 SECTION 8: LEARNING RESOURCES & REFERENCES
    ═══════════════════════════════════════════════════════
    - Research papers to read (with authors and year)
    - Online courses and tutorials
    - GitHub repositories worth exploring
    - Books and documentation
    - Communities and forums to join

    ═══════════════════════════════════════════════════════
    💡 SECTION 9: EXPERT TIPS & ADDITIONAL INSIGHTS
    ═══════════════════════════════════════════════════════
    - Pro tips specific to user's use case
    - Common mistakes to avoid
    - How to stand out / make the project impressive
    - Future extensions and improvements after initial goal

    ═══════════════════════════════════════════════════════
    ✅ SECTION 10: FINAL VERDICT & ENCOURAGEMENT
    ═══════════════════════════════════════════════════════
    - Overall feasibility score (1-10) with justification
    - Summary of the most critical next steps
    - Motivational closing remarks

    RULES:
    - Be as detailed and specific as possible in EVERY section
    - Use bullet points, sub-bullets, and clear formatting
    - Tailor EVERYTHING to the user's specific answers — do NOT give generic advice
    - Minimum length: 1500 words
    - ALWAYS call the write_text_file tool after generating the full report to save it""",
    input=conversation_history,
    tools=tools
)

# ✅ Debug: print raw output types to see what's coming back
print("Output types received:", [o.type for o in final_response.output])

report_printed = False

for output in final_response.output:
    if output.type == "function_call":
        if output.name == "write_text_file":
            args = json.loads(output.arguments)
            write_text_file(
                filename=args["filename"],
                content=args["content"]
            )
            # ✅ Also print the report from the tool call itself
            print("\n" + "="*60)
            print("COMPREHENSIVE SUMMARY REPORT")
            print("="*60)
            print(args["content"])
            report_printed = True

    elif output.type == "message":
        print("\n" + "="*60)
        print("COMPREHENSIVE SUMMARY REPORT")
        print("="*60)
        # ✅ Handle both text and object content safely
        for item in output.content:
            if hasattr(item, 'text'):
                print(item.text)
            else:
                print(item)
        report_printed = True

# ✅ Fallback: if nothing was caught, print output_text directly
if not report_printed:
    print("\n" + "="*60)
    print("COMPREHENSIVE SUMMARY REPORT")
    print("="*60)
    print(final_response.output_text)