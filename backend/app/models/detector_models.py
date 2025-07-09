import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
import spacy
import numpy as np
from typing import Dict, Any, List, Optional, Union
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
from app.utils.logger import app_logger
from app.utils.exceptions import DetectionException, ModelLoadingException

class ToxicityDetector:
    def __init__(self, model_name: str = "martin-ha/toxic-comment-model"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
    async def load_model(self):
        try:
            app_logger.info(f"Loading toxicity model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            self.is_loaded = True
            app_logger.info("Toxicity model loaded successfully")
            
        except Exception as e:
            app_logger.error(f"Failed to load toxicity model: {e}")
            raise ModelLoadingException(f"Failed to load toxicity model: {e}")
    
    async def detect(self, text: str, threshold: float = 0.7) -> Dict[str, Any]:
        try:
            if not self.is_loaded:
                await self.load_model()
            
            if not text.strip():
                return {"is_toxic": False, "confidence": 0.0, "scores": {}}
            
            results = self.pipeline(text)
            
            # Handle different model output formats
            if isinstance(results[0], list):
                scores = {item['label']: item['score'] for item in results[0]}
            else:
                scores = {results[0]['label']: results[0]['score']}
            
            # Determine toxicity
            toxic_score = scores.get('TOXIC', scores.get('toxic', 0.0))
            is_toxic = toxic_score > threshold
            
            return {
                "is_toxic": is_toxic,
                "confidence": toxic_score,
                "scores": scores,
                "threshold": threshold
            }
            
        except Exception as e:
            app_logger.error(f"Toxicity detection failed: {e}")
            raise DetectionException(f"Toxicity detection failed: {e}")

class PIIDetector:
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self.is_loaded = False
        
        # Common PII patterns
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'),
            'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        }
        
    async def load_model(self):
        try:
            app_logger.info(f"Loading PII detection model: {self.model_name}")
            
            self.nlp = spacy.load(self.model_name)
            self.is_loaded = True
            
            app_logger.info("PII detection model loaded successfully")
            
        except Exception as e:
            app_logger.error(f"Failed to load PII model: {e}")
            raise ModelLoadingException(f"Failed to load PII model: {e}")
    
    async def detect(self, text: str, sensitivity: float = 0.8) -> Dict[str, Any]:
        try:
            if not self.is_loaded:
                await self.load_model()
            
            if not text.strip():
                return {"has_pii": False, "entities": [], "patterns": []}
            
            # NER detection
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                # Only flag specific PII that looks like real personal information
                if ent.label_ == 'PERSON':
                    # Skip common names/historical figures - only flag if it looks like a real name with context
                    if len(ent.text.split()) > 1 or (len(ent.text.split()) == 1 and ent.text[0].islower()):
                        entities.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'confidence': 1.0
                        })
                elif ent.label_ in ['ORG'] and len(ent.text) > 3:
                    # Only flag organizations that look like real entities
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 1.0
                    })
                # Skip DATE, MONEY, CARDINAL - too general for PII detection
            
            # Pattern-based detection
            pattern_matches = []
            for pattern_name, pattern in self.patterns.items():
                matches = pattern.finditer(text)
                for match in matches:
                    pattern_matches.append({
                        'text': match.group(),
                        'type': pattern_name,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 1.0
                    })
            
            has_pii = len(entities) > 0 or len(pattern_matches) > 0
            
            return {
                "has_pii": has_pii,
                "entities": entities,
                "patterns": pattern_matches,
                "sensitivity": sensitivity
            }
            
        except Exception as e:
            app_logger.error(f"PII detection failed: {e}")
            raise DetectionException(f"PII detection failed: {e}")

class PromptInjectionDetector:
    def __init__(self):
        self.is_loaded = False
        self.injection_patterns = [
            r'ignore\s+previous\s+instructions',
            r'ignore\s+the\s+above',
            r'disregard\s+the\s+above',
            r'forget\s+everything\s+above',
            r'you\s+are\s+now\s+playing\s+the\s+role\s+of',
            r'pretend\s+you\s+are',
            r'act\s+as\s+if\s+you\s+are',
            r'simulate\s+being',
            r'roleplay\s+as',
            r'behave\s+like',
            r'respond\s+as\s+if\s+you\s+were',
            r'from\s+now\s+on',
            r'new\s+instruction',
            r'override\s+your\s+programming',
            r'change\s+your\s+behavior',
            r'alter\s+your\s+responses'
        ]
        
    async def load_model(self):
        try:
            app_logger.info("Loading prompt injection detector")
            
            # Compile regex patterns
            self.compiled_patterns = [
                re.compile(pattern, re.IGNORECASE) for pattern in self.injection_patterns
            ]
            
            self.is_loaded = True
            app_logger.info("Prompt injection detector loaded successfully")
            
        except Exception as e:
            app_logger.error(f"Failed to load prompt injection detector: {e}")
            raise ModelLoadingException(f"Failed to load prompt injection detector: {e}")
    
    async def detect(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        try:
            if not self.is_loaded:
                await self.load_model()
            
            if not text.strip():
                return {"is_injection": False, "confidence": 0.0, "matched_patterns": []}
            
            matched_patterns = []
            total_matches = 0
            
            for i, pattern in enumerate(self.compiled_patterns):
                matches = pattern.findall(text)
                if matches:
                    matched_patterns.append({
                        'pattern': self.injection_patterns[i],
                        'matches': matches,
                        'count': len(matches)
                    })
                    total_matches += len(matches)
            
            # Calculate confidence based on number of matches
            confidence = min(total_matches * 0.3, 1.0)
            is_injection = confidence > threshold
            
            return {
                "is_injection": is_injection,
                "confidence": confidence,
                "matched_patterns": matched_patterns,
                "threshold": threshold
            }
            
        except Exception as e:
            app_logger.error(f"Prompt injection detection failed: {e}")
            raise DetectionException(f"Prompt injection detection failed: {e}")

class TopicDetector:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
        
        # Predefined restricted topics with sample texts
        self.restricted_topics = {
            'violence': [
                "physical harm", "weapons", "assault", "fighting", "violence",
                "murder", "killing", "attack", "terrorism", "war"
            ],
            'hate_speech': [
                "racial slurs", "discrimination", "prejudice", "hatred",
                "racist", "sexist", "homophobic", "bigotry"
            ],
            'illegal_activities': [
                "drug dealing", "fraud", "hacking", "theft", "money laundering",
                "illegal substances", "criminal activities", "law breaking"
            ],
            'adult_content': [
                "explicit sexual content", "pornography", "adult material",
                "sexual acts", "inappropriate content"
            ]
        }
        
    async def load_model(self):
        try:
            app_logger.info(f"Loading topic detection model: {self.model_name}")
            
            self.model = SentenceTransformer(self.model_name)
            
            # Pre-compute embeddings for restricted topics
            self.topic_embeddings = {}
            for topic, examples in self.restricted_topics.items():
                embeddings = self.model.encode(examples)
                self.topic_embeddings[topic] = np.mean(embeddings, axis=0)
            
            self.is_loaded = True
            app_logger.info("Topic detection model loaded successfully")
            
        except Exception as e:
            app_logger.error(f"Failed to load topic detection model: {e}")
            raise ModelLoadingException(f"Failed to load topic detection model: {e}")
    
    async def detect(self, text: str, threshold: float = 0.7) -> Dict[str, Any]:
        try:
            if not self.is_loaded:
                await self.load_model()
            
            if not text.strip():
                return {"has_restricted_topic": False, "similarities": {}, "max_similarity": 0.0}
            
            # Encode input text
            text_embedding = self.model.encode([text])[0]
            
            # Calculate similarities with restricted topics
            similarities = {}
            for topic, topic_embedding in self.topic_embeddings.items():
                similarity = np.dot(text_embedding, topic_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(topic_embedding)
                )
                similarities[topic] = float(similarity)
            
            max_similarity = max(similarities.values())
            has_restricted_topic = max_similarity > threshold
            
            return {
                "has_restricted_topic": has_restricted_topic,
                "similarities": similarities,
                "max_similarity": max_similarity,
                "threshold": threshold
            }
            
        except Exception as e:
            app_logger.error(f"Topic detection failed: {e}")
            raise DetectionException(f"Topic detection failed: {e}")

class FactCheckDetector:
    def __init__(self):
        self.is_loaded = False
        self.factual_indicators = [
            'according to', 'studies show', 'research indicates',
            'statistics reveal', 'data shows', 'evidence suggests',
            'proven fact', 'scientific evidence', 'documented case'
        ]
        
    async def load_model(self):
        try:
            app_logger.info("Loading fact-check detector")
            self.is_loaded = True
            app_logger.info("Fact-check detector loaded successfully")
            
        except Exception as e:
            app_logger.error(f"Failed to load fact-check detector: {e}")
            raise ModelLoadingException(f"Failed to load fact-check detector: {e}")
    
    async def detect(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        try:
            if not self.is_loaded:
                await self.load_model()
            
            if not text.strip():
                return {"needs_fact_check": False, "confidence": 0.0, "factual_claims": []}
            
            text_lower = text.lower()
            factual_claims = []
            
            # Look for factual indicators
            for indicator in self.factual_indicators:
                if indicator in text_lower:
                    factual_claims.append(indicator)
            
            # Simple heuristic: if text contains numbers and factual indicators
            has_numbers = bool(re.search(r'\d+', text))
            has_factual_language = len(factual_claims) > 0
            
            confidence = 0.0
            if has_numbers and has_factual_language:
                confidence = 0.8
            elif has_numbers or has_factual_language:
                confidence = 0.4
            
            needs_fact_check = confidence > threshold
            
            return {
                "needs_fact_check": needs_fact_check,
                "confidence": confidence,
                "factual_claims": factual_claims,
                "has_numbers": has_numbers,
                "threshold": threshold
            }
            
        except Exception as e:
            app_logger.error(f"Fact-check detection failed: {e}")
            raise DetectionException(f"Fact-check detection failed: {e}")

class SpamDetector:
    def __init__(self):
        self.is_loaded = False
        self.spam_indicators = [
            'click here', 'buy now', 'limited time', 'act now',
            'urgent', 'congratulations', 'winner', 'free money',
            'make money fast', 'work from home', 'get rich quick'
        ]
        
    async def load_model(self):
        try:
            app_logger.info("Loading spam detector")
            self.is_loaded = True
            app_logger.info("Spam detector loaded successfully")
            
        except Exception as e:
            app_logger.error(f"Failed to load spam detector: {e}")
            raise ModelLoadingException(f"Failed to load spam detector: {e}")
    
    async def detect(self, text: str, threshold: float = 0.6) -> Dict[str, Any]:
        try:
            if not self.is_loaded:
                await self.load_model()
            
            if not text.strip():
                return {"is_spam": False, "confidence": 0.0, "spam_indicators": []}
            
            text_lower = text.lower()
            found_indicators = []
            
            # Check for spam indicators
            for indicator in self.spam_indicators:
                if indicator in text_lower:
                    found_indicators.append(indicator)
            
            # Additional spam characteristics
            excessive_caps = len(re.findall(r'[A-Z]', text)) / len(text) > 0.3 if text else False
            excessive_punctuation = len(re.findall(r'[!?]{2,}', text)) > 0
            
            # Calculate confidence
            confidence = 0.0
            confidence += len(found_indicators) * 0.2
            if excessive_caps:
                confidence += 0.3
            if excessive_punctuation:
                confidence += 0.2
            
            confidence = min(confidence, 1.0)
            is_spam = confidence > threshold
            
            return {
                "is_spam": is_spam,
                "confidence": confidence,
                "spam_indicators": found_indicators,
                "excessive_caps": excessive_caps,
                "excessive_punctuation": excessive_punctuation,
                "threshold": threshold
            }
            
        except Exception as e:
            app_logger.error(f"Spam detection failed: {e}")
            raise DetectionException(f"Spam detection failed: {e}")