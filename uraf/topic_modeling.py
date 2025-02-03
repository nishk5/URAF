from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np

class TopicModeling:
    """
    Topic modeling using BERTopic with real-time analysis capabilities.
    Extracts high-level themes from LLM responses and supports incremental updates.
    """

    def __init__(self, min_topic_size=3):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Initialize BERTopic with parameters optimized for LLM responses
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            min_topic_size=min_topic_size,
            verbose=True
        )
        self.topic_cache = {}  # Cache for incremental updates

    def extract_topics(self, responses, batch_size=32):
        """
        Extracts key themes from multiple LLM responses with batched processing.
        
        Args:
            responses: List of LLM-generated texts
            batch_size: Size of batches for processing large sets of responses
        
        Returns:
            Dictionary containing:
            - topics: List of topic IDs
            - topic_info: DataFrame with topic details
            - representative_docs: Most representative document per topic
        """
        # Process in batches for memory efficiency
        topics, probs = [], []
        for i in range(0, len(responses), batch_size):
            batch = responses[i:i + batch_size]
            batch_topics, batch_probs = self.topic_model.fit_transform(batch)
            topics.extend(batch_topics)
            probs.extend(batch_probs)

        # Get topic information
        topic_info = self.topic_model.get_topic_info()
        
        # Find most representative document per topic
        topic_docs = {}
        for topic_id in set(topics):
            if topic_id != -1:  # Skip outlier topic
                topic_docs[topic_id] = self._get_representative_doc(
                    [doc for doc, t in zip(responses, topics) if t == topic_id]
                )

        return {
            "topics": topics,
            "topic_info": topic_info,
            "representative_docs": topic_docs
        }

    def update_topics(self, new_responses):
        """
        Real-time update of topics with new responses.
        Efficiently updates the topic model without full retraining.
        
        Args:
            new_responses: List of new LLM responses to analyze
        
        Returns:
            Dictionary with updated topic information
        """
        # Update the topic model incrementally
        self.topic_model.partial_fit(new_responses)
        
        # Get updated topic information
        updated_info = self.topic_model.get_topic_info()
        
        # Cache the results for future updates
        self.topic_cache.update({
            "last_update": len(new_responses),
            "topic_info": updated_info
        })
        
        return {
            "updated_topics": updated_info,
            "new_topics": self._detect_new_topics(updated_info)
        }

    def _get_representative_doc(self, docs):
        """
        Find the most representative document for a topic using embeddings.
        """
        if not docs:
            return None
            
        # Get embeddings for all documents
        embeddings = self.embedding_model.encode(docs, convert_to_tensor=True)
        
        # Calculate centroid
        centroid = embeddings.mean(dim=0)
        
        # Find document closest to centroid
        similarities = embeddings @ centroid
        most_representative_idx = similarities.argmax().item()
        
        return docs[most_representative_idx]

    def _detect_new_topics(self, updated_info):
        """
        Detect newly emerged topics after an update.
        """
        if "topic_info" not in self.topic_cache:
            return []
            
        old_topics = set(self.topic_cache["topic_info"]["Topic"])
        new_topics = set(updated_info["Topic"])
        
        return list(new_topics - old_topics)
