"""
SOC20 Classifier - Local Version for Winston (Slurm)

This module provides SOC20 classification using local models:
- Qwen2.5 models (7B/14B/32B/72B) for chat/classification
- sentence-transformers for embeddings
- ChromaDB for vector storage

Designed for batch processing on Winston cluster via Slurm.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SOCClassifier:
    """
    Local SOC20 classifier using open models.
    """

    def __init__(
        self,
        chat_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        chroma_path: str = "/data/spack/users/hod123/chroma",
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize the SOC classifier.

        Args:
            chat_model_name: HuggingFace model name for chat (e.g., "Qwen/Qwen2.5-7B-Instruct")
            embedding_model_name: sentence-transformers model name
            chroma_path: Path to ChromaDB persistence directory
            device: Device to use ("auto", "cuda", "cpu")
            load_in_8bit: Whether to load chat model in 8-bit quantization
            load_in_4bit: Whether to load chat model in 4-bit quantization
        """
        self.chat_model_name = chat_model_name
        self.embedding_model_name = embedding_model_name
        self.chroma_path = chroma_path

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"CUDA devices available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        # Load models
        logger.info(f"Loading chat model: {chat_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(chat_model_name)

        # Configure quantization if requested
        model_kwargs = {
            "device_map": "auto" if self.device == "cuda" else None,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            logger.info("Loading model in 8-bit mode")
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            logger.info("Loading model in 4-bit mode")

        self.chat_model = AutoModelForCausalLM.from_pretrained(
            chat_model_name,
            **model_kwargs
        )

        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            device=self.device
        )

        # Initialize ChromaDB
        logger.info(f"Connecting to ChromaDB at: {chroma_path}")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        # Load prompts from static directory
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from static directory."""
        prompts = {}
        static_dir = Path(__file__).parent / "static"

        if not static_dir.exists():
            logger.warning(f"Static directory not found: {static_dir}")
            return prompts

        for prompt_file in static_dir.glob("*.txt"):
            prompts[prompt_file.name] = prompt_file.read_text("utf-8")
            logger.info(f"Loaded prompt: {prompt_file.name}")

        return prompts

    @lru_cache(maxsize=2000)
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using sentence-transformers.
        Results are cached for efficiency.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding
        """
        try:
            embedding = self.embedding_model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def get_shortlist(
        self,
        text: str,
        collection_name: str = "job_titles_4d",
        k: int = 10
    ) -> str:
        """
        Query ChromaDB for top-k similar SOC codes.

        Args:
            text: Query text (e.g., job title)
            collection_name: ChromaDB collection to query
            k: Number of results to return

        Returns:
            Newline-separated string of candidate SOC codes
        """
        try:
            # Get or create collection
            collection = self.chroma_client.get_or_create_collection(collection_name)

            # Generate embedding
            embedding = self.get_embedding(text)

            # Query ChromaDB
            results = collection.query(
                query_embeddings=[embedding],
                n_results=k,
                include=["metadatas", "documents", "distances"]
            )

            # Check if results are valid
            if not results or not results['ids'] or len(results['ids'][0]) == 0:
                logger.warning(f"Empty results from ChromaDB for query: {text}")
                return ""

            # Extract results
            ids = results['ids'][0]
            metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(ids)
            documents = results['documents'][0] if results['documents'] else ids

            if collection_name == "job_titles_4d":
                # Extract code descriptor (desc) for each match, remove duplicates
                unique_cands = list(
                    dict.fromkeys([
                        meta.get("desc", doc)
                        for meta, doc in zip(metadatas, documents)
                    ])
                )
                cands = "\n".join(unique_cands)
            else:
                # For soc4d, use document IDs directly
                cands = "\n".join(ids)

            return cands

        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            raise

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.01,
        top_p: float = 1.0,
    ) -> str:
        """
        Generate a response using the chat model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text response
        """
        try:
            # Format messages using the chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096  # Adjust based on model's context window
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.chat_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the generated tokens (not the prompt)
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def classify(
        self,
        init_q: str,
        init_ans: str,
        sys_prompt: str = "force_classify.txt",
        k: int = 10,
        collection_name: str = "job_titles_4d",
        soc_cands: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Classify a job title into SOC code (forced classification, no follow-ups).

        Args:
            init_q: Initial question (e.g., "What was your main job title?")
            init_ans: User's answer (job title)
            sys_prompt: System prompt filename or full text
            k: Number of candidate codes to retrieve
            collection_name: ChromaDB collection to query
            soc_cands: Pre-computed candidates (optional, bypasses retrieval)

        Returns:
            Dict with soc_code, soc_desc, soc_conf, soc_followup, soc_cands, response
        """
        # Get system prompt
        if sys_prompt.endswith(".txt"):
            sys_prompt_text = self.prompts.get(sys_prompt, "")
            if not sys_prompt_text:
                raise ValueError(f"Prompt file not found: {sys_prompt}")
        else:
            sys_prompt_text = sys_prompt

        # Get candidate SOC codes if not provided
        if soc_cands is None:
            job_str = f"Job title: '{init_ans}'"
            soc_cands = self.get_shortlist(job_str, collection_name, k)

        # Format system prompt with candidates
        sys_prompt_text = sys_prompt_text.format(K_soc=soc_cands)

        # Build message list
        messages = [
            {"role": "system", "content": sys_prompt_text},
            {"role": "assistant", "content": init_q},
            {"role": "user", "content": init_ans},
        ]

        # Generate response
        response = self.generate_response(messages)

        # Parse response
        result = self._parse_classify_response(response, soc_cands)

        return result

    def followup(
        self,
        init_q: str,
        init_ans: str,
        sys_prompt: str = "followup_prompt.txt",
        k: int = 10,
        collection_name: str = "job_titles_4d",
        soc_cands: Optional[str] = None,
        additional_qs: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, str]:
        """
        Classify with potential follow-up questions.

        Args:
            init_q: Initial question
            init_ans: User's answer
            sys_prompt: System prompt filename or full text
            k: Number of candidate codes to retrieve
            collection_name: ChromaDB collection to query
            soc_cands: Pre-computed candidates (optional)
            additional_qs: List of (question, answer) tuples from previous follow-ups

        Returns:
            Dict with soc_code, soc_desc, soc_conf, followup, soc_cands
        """
        # Get system prompt
        if sys_prompt.endswith(".txt"):
            sys_prompt_text = self.prompts.get(sys_prompt, "")
            if not sys_prompt_text:
                raise ValueError(f"Prompt file not found: {sys_prompt}")
        else:
            sys_prompt_text = sys_prompt

        # Handle REQUERY case
        if soc_cands == "REQUERY" and additional_qs:
            job_str = f"Job title: '{additional_qs[0][1]}'"
            soc_cands = self.get_shortlist(job_str, collection_name, k)
        elif soc_cands is None:
            job_str = f"Job title: '{init_ans}'"
            soc_cands = self.get_shortlist(job_str, collection_name, k)

        # Format system prompt with candidates
        sys_prompt_text = sys_prompt_text.format(K_soc=soc_cands)

        # Build message list
        messages = [
            {"role": "system", "content": sys_prompt_text},
            {"role": "assistant", "content": init_q},
            {"role": "user", "content": init_ans},
        ]

        # Add additional Q&A pairs
        if additional_qs:
            for q, a in additional_qs:
                messages.append({"role": "assistant", "content": q})
                messages.append({"role": "user", "content": a})

        # Generate response
        response = self.generate_response(messages, temperature=0.01, top_p=1.0)

        # Parse response
        result = self._parse_followup_response(response, soc_cands)

        return result

    def _parse_classify_response(self, response: str, soc_cands: str) -> Dict[str, str]:
        """Parse classification response."""
        if response.startswith("CGPT587:"):
            try:
                soc_code = re.findall(r"(?<=CGPT587:\s)\d{4}", response)[0]
                soc_desc = re.findall(
                    r"(?<=CGPT587:\s\d{4}\s-\s)(.*?)(?=;\sCONFIDENCE:)", response
                )[0]
                soc_conf = re.findall(r"(?<=CONFIDENCE:\s)\d+", response)[0]
                soc_followup = re.findall(r"(?<=FOLLOWUP:\s)(TRUE|FALSE)", response)[0]
            except (IndexError, ValueError):
                soc_code = "ERROR"
                soc_desc = "ERROR"
                soc_conf = "ERROR"
                soc_followup = "ERROR"
        else:
            soc_code = "NONE"
            soc_desc = "NONE"
            soc_conf = "NONE"
            soc_followup = "NONE"

        return {
            "soc_code": soc_code,
            "soc_desc": soc_desc,
            "soc_conf": soc_conf,
            "soc_followup": soc_followup,
            "soc_cands": soc_cands.replace("\n", ", "),
            "response": response,
        }

    def _parse_followup_response(self, response: str, soc_cands: str) -> Dict[str, str]:
        """Parse follow-up response."""
        if len(re.findall("CGPT587", response)) > 0:
            try:
                soc_code = re.findall(r"(?<=CGPT587:\s)\d{4}", response)[0]
                soc_desc = re.findall(
                    r"(?<=CGPT587:\s\d{4}\s-\s).*(?=\s\(\d+\)$)", response
                )[0]
                soc_conf = re.findall(r"\d+(?=\)$)", response)[0]
            except (IndexError, ValueError):
                soc_code = "ERROR"
                soc_desc = "ERROR"
                soc_conf = "ERROR"
        elif len(re.findall("CGPT631:", response)) > 0:
            soc_code = "NONE"
            soc_desc = "NONE"
            soc_conf = "NONE"
            soc_cands = "REQUERY"
            response = re.sub("CGPT631: ", "", response)
        else:
            soc_code = "NONE"
            soc_desc = "NONE"
            soc_conf = "NONE"

        return {
            "soc_code": soc_code,
            "soc_desc": soc_desc,
            "soc_conf": soc_conf,
            "followup": response,
            "soc_cands": soc_cands,
        }


if __name__ == "__main__":
    # Example usage
    classifier = SOCClassifier(
        chat_model_name="Qwen/Qwen2.5-7B-Instruct",
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        chroma_path="/data/spack/users/hod123/chroma",
    )

    # Test classification
    result = classifier.classify(
        init_q="What was your main job title in the past week?",
        init_ans="software developer",
        k=10
    )

    print("Classification result:")
    print(f"SOC Code: {result['soc_code']}")
    print(f"SOC Description: {result['soc_desc']}")
    print(f"Confidence: {result['soc_conf']}")
    print(f"Response: {result['response']}")
