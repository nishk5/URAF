from flashtext import KeywordProcessor
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from .topic_modeling import TopicModeling
from .summary_comparator import SummaryComparator

class ResponseProcessor:
    """
    Modern response processor using lightweight, LLM-friendly NLP stack:
    - FlashText for fast keyword extraction
    - KeyBERT for embeddings-based keyphrase extraction
    - Sentence-Transformers for semantic similarity
    - BERTopic for dynamic topic modeling
    """

    def __init__(self):
        # Core NLP components
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        self.bert_model = KeyBERT()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Advanced analysis components
        self.topic_model = TopicModeling(min_topic_size=2)  # Smaller size for individual responses
        self.summary_comparator = SummaryComparator()
        
        # Common entities we want to track
        self.common_entities = [
            "person", "organization", "location", "date", "time",
            "money", "percent", "product", "event", "technology"
        ]
        self.keyword_processor.add_keywords_from_list(self.common_entities)
        
        # Response cache for comparative analysis
        self.response_cache = []
        self.max_cache_size = 10

    def process(self, text, compare_with_history=True):
        """
        Enhanced response processing with topic modeling and historical comparison.
        
        Args:
            text: Input text to process
            compare_with_history: Whether to compare with previous responses
            
        Returns:
            Dictionary with extracted information and comparative analysis
        """
        # Basic processing
        keyphrases = self.bert_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words=None,
            use_maxsum=True,
            top_n=5
        )
        
        # Entity extraction
        entities = [(keyword, "ENTITY") for keyword in self.keyword_processor.extract_keywords(text)]
        
        # Generate summary using sentence embeddings
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if sentences:
            # Compute sentence embeddings
            embeddings = self.embedding_model.encode(sentences, convert_to_tensor=True)

            # Compute similarity across sentences
            similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

            # Rank sentences by average similarity to others
            mean_sim_scores = similarity_matrix.mean(dim=1)
            top_idx = mean_sim_scores.argmax().item()

            # Select the most representative sentence
            summary = sentences[top_idx]
        else:
            summary = text

        # Extract topics from the current response
        current_topics = self.topic_model.extract_topics([text])
        
        # Comparative analysis with history
        historical_comparison = None
        if compare_with_history and self.response_cache:
            historical_comparison = self.summary_comparator.compare_summaries(
                summaries=[text] + self.response_cache[-3:],  # Compare with last 3 responses
                threshold=0.75
            )

        # Update response cache
        self.response_cache.append(text)
        if len(self.response_cache) > self.max_cache_size:
            self.response_cache.pop(0)

        # Combine all analysis
        result = {
            "raw_text": text,
            "summary": summary,
            "entities": entities,
            "keyphrases": [kp for kp, score in keyphrases],
            "topics": current_topics["topic_info"].to_dict() if len(current_topics["topics"]) > 0 else {},
            "historical_comparison": historical_comparison
        }

        return result

    def batch_process(self, texts):
        """
        Process multiple responses together for comparative analysis.
        
        Args:
            texts: List of responses to analyze
            
        Returns:
            Dictionary with individual and comparative analysis
        """
        # Individual processing
        individual_results = [self.process(text, compare_with_history=False) for text in texts]
        
        # Collective topic analysis
        collective_topics = self.topic_model.extract_topics(texts)
        
        # Cross-response comparison
        comparison = self.summary_comparator.compare_summaries(
            summaries=[result["summary"] for result in individual_results],
            threshold=0.75
        )
        
        return {
            "individual_results": individual_results,
            "collective_topics": collective_topics["topic_info"].to_dict(),
            "cross_response_comparison": comparison
        }
