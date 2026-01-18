import os
import requests
import numpy as np
from dataclasses import dataclass
from app.config_loader import load_settings
from app.rag_engine import SmolFusionRAG  # Import the hybrid RAG engine

# Load application settings
settings = load_settings()

# LLM API configuration
LLM_URL    = settings["llm"].get("url") or os.getenv("LLM_URL", "http://localhost:8080")
MODEL_NAME = settings["llm"]["model"]
TEMP       = settings["llm"]["temperature"]
MAXTOK     = settings["llm"]["max_tokens"]
ROBOT_NAME = settings.get("robot_name", "Igris")
LANG       = settings.get("language", "Korean")

# Initialize RAG Engine with domain-specific knowledge (e.g., Robot Manual, HRI Protocols)
# In a real scenario, this could be loaded from a JSON or Markdown file.
knowledge_base = [
    "Igris is a 1.7B parameter humanoid robot optimized for real-time HRI.",
    "When battery is below 20%, Igris must navigate to the charging station.",
    "Igris uses SmolVLM for visual perception and Qwen for language reasoning.",
    "The robot has 6 degrees of freedom in each arm for precise manipulation."
]
rag_engine = SmolFusionRAG(knowledge_base)

@dataclass
class ChatConfig:
    """
    Configuration for generating a chat response with RAG context.
    """
    prompt: str
    session_id: str = "default"
    max_tokens: int = MAXTOK
    persona: str = ROBOT_NAME
    language: str = LANG

def generate_response_llm_rag(config: ChatConfig) -> str:
    """
    Retrieves context using RAG and sends an enriched prompt to the LLM API.
    
    This method compensates for the limited parameter size of the 1.7B model
    by injecting factual grounding directly into the system prompt.
    """
    
    # [Step 1] Retrieval: Get the top-2 most relevant documents
    retrieved_context = rag_engine.search(config.prompt, top_k=2)
    context_text = "\n".join(retrieved_context)
    
    # [Step 2] Augmentation: Inject retrieved knowledge into the system prompt
    # This 'Grounding' technique prevents hallucinations in small-scale models.
    enriched_system_prompt = (
        f"You are {config.persona}, a helpful humanoid robot. Reply in {config.language}. "
        f"Be concise and friendly.\n\n"
        f"### Reference Knowledge (Grounding):\n{context_text}\n\n"
        f"Please answer the user's question based on the reference knowledge provided above."
    )

    # [Step 3] Generation: Standard OpenAI-style API call to llama-server
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": enriched_system_prompt},
            {"role": "user", "content": config.prompt}
        ],
        "max_tokens": int(config.max_tokens),
        "temperature": float(TEMP),
    }

    try:
        r = requests.post(f"{LLM_URL}/v1/chat/completions", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error in RAG-LLM Pipeline: {str(e)}"
