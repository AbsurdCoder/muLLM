"""
Evaluation metrics and testing utilities for language models.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
import json
import os
from pathlib import Path
import logging


class ModelTester:
    """Class for testing and evaluating language models."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the tester.
        
        Args:
            model: Model to test
            tokenizer: Tokenizer for encoding/decoding text
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Text prompt to start generation from
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to consider for sampling
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated text sequences
        """
        self.model.eval()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        
        # Add BOS token if not present
        if input_ids[0] != self.tokenizer.token_to_id["[BOS]"]:
            input_ids = [self.tokenizer.token_to_id["[BOS]"]] + input_ids
        
        # Convert to tensor and repeat for multiple sequences
        input_ids = torch.tensor([input_ids] * num_return_sequences, dtype=torch.long).to(self.device)
        
        # Generate text
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )
        
        # Decode generated sequences
        generated_texts = []
        for ids in output_ids:
            text = self.tokenizer.decode(ids.tolist())
            generated_texts.append(text)
        
        return generated_texts
    
    def calculate_perplexity(self, texts: List[str], max_length: int = 512) -> float:
        """
        Calculate perplexity on a set of texts.
        
        Args:
            texts: List of text samples
            max_length: Maximum sequence length
            
        Returns:
            Average perplexity across all texts
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)  # Ignore padding index
        with torch.no_grad():
            for text in tqdm(texts, desc="Calculating perplexity"):
                # Tokenize text
                token_ids = self.tokenizer.encode(text)
                # Truncate if necessary
                if len(token_ids) > max_length - 2:  # -2 for BOS and EOS tokens
                    token_ids = token_ids[:max_length - 2]
                
                # Add special tokens
                token_ids = [self.tokenizer.token_to_id["[BOS]"]] + token_ids + [self.tokenizer.token_to_id["[EOS]"]]
                
                # Create input and target tensors
                input_ids = torch.tensor(token_ids[:-1], dtype=torch.long).unsqueeze(0).to(self.device)
                target_ids = torch.tensor(token_ids[1:], dtype=torch.long).to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids)
                
                # Calculate loss
                logits = outputs.squeeze(0)  # [seq_len, vocab_size]
                loss = criterion(logits, target_ids)
                
                # Update metrics
                total_loss += loss.item()
                total_tokens += len(target_ids)

        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def evaluate_accuracy(
        self,
        prompt_completion_pairs: List[Tuple[str, str]],
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50
    ) -> Dict[str, float]:
        """
        Evaluate model accuracy on prompt-completion pairs.
        
        Args:
            prompt_completion_pairs: List of (prompt, expected_completion) tuples
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to consider for sampling
            
        Returns:
            Dictionary with accuracy metrics
        """
        self.model.eval()
        
        exact_matches = 0
        bleu_scores = []
        rouge_scores = []
        
        try:
            from nltk.translate.bleu_score import sentence_bleu
            from rouge import Rouge
            rouge = Rouge()
            nltk_available = True
        except ImportError:
            self.logger.warning("NLTK and/or Rouge not available. Only exact match will be calculated.")
            nltk_available = False
        
        for prompt, expected in tqdm(prompt_completion_pairs, desc="Evaluating accuracy"):
            # Generate completion
            generated_texts = self.generate_text(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                num_return_sequences=1
            )
            generated = generated_texts[0]
            
            # Check for exact match
            if generated.strip() == expected.strip():
                exact_matches += 1
            
            # Calculate BLEU and ROUGE scores if available
            if nltk_available:
                # BLEU score
                reference = [expected.strip().split()]
                candidate = generated.strip().split()
                bleu = sentence_bleu(reference, candidate)
                bleu_scores.append(bleu)
                
                # ROUGE score
                try:
                    rouge_score = rouge.get_scores(generated.strip(), expected.strip())[0]
                    rouge_scores.append(rouge_score["rouge-l"]["f"])
                except:
                    # Skip if ROUGE calculation fails (e.g., empty strings)
                    pass
        
        # Calculate metrics
        exact_match_accuracy = exact_matches / len(prompt_completion_pairs)
        
        metrics = {
            "exact_match": exact_match_accuracy
        }
        
        if nltk_available and bleu_scores:
            metrics["bleu"] = sum(bleu_scores) / len(bleu_scores)
        
        if nltk_available and rouge_scores:
            metrics["rouge_l"] = sum(rouge_scores) / len(rouge_scores)
        
        return metrics
    
    def save_test_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save test results to a file.
        
        Args:
            results: Dictionary with test results
            output_path: Path to save results to
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Test results saved to {output_path}")
    
    def run_comprehensive_test(
        self,
        test_texts: List[str],
        prompt_completion_pairs: List[Tuple[str, str]],
        output_dir: str = "./test_results",
        max_length: int = 100
    ) -> Dict[str, Any]:
        """
        Run a comprehensive test suite on the model.
        
        Args:
            test_texts: List of text samples for perplexity calculation
            prompt_completion_pairs: List of (prompt, expected_completion) tuples for accuracy evaluation
            output_dir: Directory to save test results
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with all test results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info("Starting comprehensive model testing...")
        
        # Calculate perplexity
        self.logger.info("Calculating perplexity...")
        perplexity = self.calculate_perplexity(test_texts, max_length=max_length)
        self.logger.info(f"Perplexity: {perplexity:.2f}")
        
        # Evaluate accuracy
        self.logger.info("Evaluating accuracy...")
        accuracy_metrics = self.evaluate_accuracy(prompt_completion_pairs, max_length=max_length)
        self.logger.info(f"Accuracy metrics: {accuracy_metrics}")
        
        # Generate sample texts
        self.logger.info("Generating sample texts...")
        sample_prompts = [pair[0] for pair in prompt_completion_pairs[:5]]
        sample_generations = []
        
        for prompt in sample_prompts:
            generated = self.generate_text(prompt, max_length=max_length)
            sample_generations.append({
                "prompt": prompt,
                "generated": generated[0]
            })
        
        # Compile all results
        results = {
            "perplexity": perplexity,
            "accuracy": accuracy_metrics,
            "sample_generations": sample_generations,
            "model_info": {
                "parameter_count": self.model.get_parameter_count(),
                "config": self.model.config
            }
        }
        
        # Save results
        output_path = os.path.join(output_dir, "test_results.json")
        self.save_test_results(results, output_path)
        
        return results
