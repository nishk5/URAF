import evaluate
import sentence_transformers
import re
from loguru import logger
from uraf.scorer import URAFScorer

class LLMResponseEvaluator:
    """
    Evaluates responses based on structured reasoning (URAF), semantic similarity, and NLP metrics.
    """

    def __init__(self):
        logger.info("🔍 Initializing LLMResponseEvaluator...")
        self.uraf_scorer = URAFScorer()
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.similarity_model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
        self.reference_text = ""

    async def evaluate_response(self, response_text):
        """
        Evaluates an LLM response using multiple metrics.
        Handles both 1D and 2D tensors for similarity calculation.
        """
        try:
            if not response_text:
                logger.error("❌ Empty response text")
                return 0.0
                
            # Check for required sections
            required_sections = ["Understanding:", "Reasoning Pathway:", "Final Synthesis:"]
            has_structure = all(section in response_text for section in required_sections)
            
            if not has_structure:
                logger.warning("⚠️ Response does not contain the expected structure. URAF score may be inaccurate.")
            
            # Clean up response text
            response_text = re.sub(r'<\|.*?\|>', '', response_text)
            response_text = response_text.strip()
            
            # Get embeddings
            embedding = await self.get_embeddings([self.reference_text, response_text])
            
            # Ensure embeddings are 2D for matmul
            if embedding[0].dim() == 1:
                embedding = [e.unsqueeze(0) for e in embedding]
            
            # Calculate similarity using matrix multiplication
            similarity_score = (embedding[0] @ embedding[1].mT).cpu().item()
            
            # Calculate ROUGE scores
            rouge_scores = self.rouge.compute(
                predictions=[response_text],
                references=[[response_text]],  # Self-reference for coherence
            )
            
            # Extract ROUGE-L F1 score
            rouge_l = rouge_scores["rougeL"]
            
            # Calculate BLEU score
            bleu_score = self.bleu.compute(
                predictions=[response_text],
                references=[[response_text]],  # Self-reference for coherence
                max_order=4
            )["bleu"]
            
            # Combine metrics into final score
            structure_score = 1.0 if has_structure else 0.5
            content_score = (
                similarity_score * 0.4 +
                rouge_l * 0.2 +
                bleu_score * 0.1
            )
            
            if content_score == 0:
                logger.error("❌ Division by zero")
                return 0.0
            
            final_score = (structure_score + content_score) * 5  # Scale to 0-10
            
            logger.info(f"📊 URAF Evaluation Scores:")
            logger.info(f"Structure Score: {structure_score:.2f}")
            logger.info(f"Content Score: {content_score:.2f}")
            logger.info(f"Final Score: {final_score:.2f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"❌ Error in evaluation: {str(e)}")
            return 0.0

    async def get_embeddings(self, texts):
        return self.similarity_model.encode(texts, convert_to_tensor=True)
