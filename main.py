import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.colab import userdata

# --- SETUP & AUTHENTICATION ---
# Ensure your Hugging Face token is stored in Colab Secrets as 'HF_TOKEN'
try:
    hf_access_token = userdata.get('HF_TOKEN')
except Exception:
    hf_access_token = None
    print("Notice: No HF_TOKEN found. Defaulting to public access.")

def initialize_ai_model(model_id="google/gemma-2b-it"):
    """Loads the tokenizer and model with 4-bit quantization for efficiency."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_access_token
    )
    return tokenizer, model

# Initialize components
tokenizer, model = initialize_ai_model()

# --- CORE LOGIC ---
def craft_interview_questions(candidate_bio, role_details):
    """
    Analyzes candidate experience and job specs to produce 
    curated technical and behavioral questions.
    """
    
    instruction_prompt = f"""
    Acting as a Senior Hiring Manager, analyze the following candidate profile and job description.
    Generate a list of 10 targeted interview questions (5 Technical, 5 Soft Skills/Behavioral).
    
    Criteria:
    - Technical: Must be appropriate for an intern level but challenge their specific project experience.
    - Behavioral: Use the STAR method logic to probe for past performance.

    CANDIDATE PROFILE:
    {candidate_bio}

    JOB DESCRIPTION:
    {role_details}

    REQUIRED OUTPUT FORMAT:
    ## Technical Assessment
    1-5 questions here...

    ## Behavioral Assessment
    1-5 questions here...
    """

    chat_structure = [{"role": "user", "content": instruction_prompt}]
    
    # Process inputs for the model
    input_tokens = tokenizer.apply_chat_template(
        chat_structure, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    raw_output = model.generate(
        input_tokens,
        max_new_tokens=700,
        temperature=0.75,
        do_sample=True
    )

    # Decode only the newly generated text
    response = tokenizer.decode(raw_output[0][len(input_tokens[0]):], skip_special_tokens=True)
    return response.strip()

---

# --- TEST DATA & EXECUTION ---

# Dataset of candidate profiles and roles
interview_scenarios = [
    {
        "label": "AI/Data Science Track",
        "bio": "- Skills: Python, Pandas, SQL\n- Background: 2nd Year CS Student\n- Key Project: Regression analysis on housing prices.",
        "job": "- Role: Junior Data Analyst\n- Focus: Cleaning datasets and predictive modeling."
    },
    {
        "label": "Web Development Track",
        "bio": "- Skills: JavaScript, React, Node.js\n- Background: Coding Bootcamp Grad\n- Key Project: Full-stack To-Do list with user auth.",
        "job": "- Role: Frontend Intern\n- Focus: Building responsive UI components and API integration."
    }
]

# Run the generator
for scenario in interview_scenarios:
    print(f"\n{'='*20}")
    print(f"GENERATING FOR: {scenario['label']}")
    print(f"{'='*20}")
    
    result = craft_interview_questions(scenario['bio'], scenario['job'])
    print(result)