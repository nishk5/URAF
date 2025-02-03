from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict, Any

class SummaryComparator:
    """
    Advanced summary comparison for multiple LLM outputs.
    Analyzes coherence, semantic similarity, and content preservation
    across different LLM-generated summaries.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def compare_summaries(self, 
                         summaries: List[str], 
                         original_text: str = None,
                         threshold: float = 0.75) -> Dict[str, Any]:
        """
        Comprehensive comparison of multiple LLM-generated summaries.
        
        Args:
            summaries: List of summaries from different LLMs
            original_text: Optional original text for content preservation analysis
            threshold: Similarity threshold for identifying significant differences
        
        Returns:
            Dictionary containing:
            - similarity_matrix: Pairwise similarities between summaries
            - coherence_scores: Coherence score for each summary
            - content_preservation: How well each summary preserves original content
            - consensus_summary: Most representative summary
        """
        # Encode all summaries
        summary_embeddings = self.model.encode(summaries, convert_to_tensor=True)
        
        # Compute pairwise similarities
        similarity_matrix = util.pytorch_cos_sim(summary_embeddings, summary_embeddings)
        
        # Calculate coherence scores
        coherence_scores = self._calculate_coherence(summaries)
        
        # Content preservation analysis
        content_preservation = None
        if original_text:
            content_preservation = self._analyze_content_preservation(
                summaries, original_text
            )
        
        # Find consensus summary
        consensus_idx = self._find_consensus_summary(similarity_matrix)
        
        # Identify significant differences
        differences = self._identify_differences(
            summaries, similarity_matrix, threshold
        )
        
        return {
            "similarity_matrix": similarity_matrix.tolist(),
            "coherence_scores": coherence_scores,
            "content_preservation": content_preservation,
            "consensus_summary": summaries[consensus_idx],
            "significant_differences": differences
        }
    
    def _calculate_coherence(self, summaries: List[str]) -> List[float]:
        """
        Calculate coherence score for each summary based on sentence-level similarity.
        """
        coherence_scores = []
        
        for summary in summaries:
            # Split into sentences and encode
            sentences = [s.strip() for s in summary.split('.') if s.strip()]
            if len(sentences) < 2:
                coherence_scores.append(1.0)  # Perfect coherence for single sentence
                continue
                
            # Calculate sentence embeddings
            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
            
            # Calculate adjacent sentence similarities
            similarities = util.pytorch_cos_sim(
                sentence_embeddings[:-1], 
                sentence_embeddings[1:]
            )
            
            # Average similarity as coherence score
            coherence_scores.append(float(similarities.mean()))
            
        return coherence_scores
    
    def _analyze_content_preservation(self, 
                                    summaries: List[str], 
                                    original_text: str) -> List[float]:
        """
        Analyze how well each summary preserves the original content.
        """
        # Encode original text
        original_embedding = self.model.encode(original_text, convert_to_tensor=True)
        
        # Encode and compare each summary
        preservation_scores = []
        for summary in summaries:
            summary_embedding = self.model.encode(summary, convert_to_tensor=True)
            score = float(util.pytorch_cos_sim(original_embedding, summary_embedding))
            preservation_scores.append(score)
            
        return preservation_scores
    
    def _find_consensus_summary(self, similarity_matrix) -> int:
        """
        Find the summary that best represents the consensus across all summaries.
        """
        # Calculate average similarity of each summary to all others
        mean_similarities = similarity_matrix.mean(dim=1)
        return int(mean_similarities.argmax())
    
    def _identify_differences(self, 
                            summaries: List[str], 
                            similarity_matrix,
                            threshold: float) -> List[Dict[str, Any]]:
        """
        Identify significant differences between summaries.
        """
        differences = []
        n_summaries = len(summaries)
        
        for i in range(n_summaries):
            for j in range(i + 1, n_summaries):
                similarity = float(similarity_matrix[i, j])
                if similarity < threshold:
                    differences.append({
                        "summary1_idx": i,
                        "summary2_idx": j,
                        "similarity": similarity,
                        "summaries": [summaries[i], summaries[j]]
                    })
                    
        return differences
