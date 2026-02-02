
import random
import nltk
from typing import List, Dict, Optional, Callable
from nltk.corpus import wordnet
from ..utils.logger import logger

class SyntheticQAGenerator:
    def __init__(self):
        self._initialized = False

    def _ensure_nltk(self, on_progress: Optional[Callable[[str], None]] = None):
        if self._initialized:
            return
        
        logger.info("Verifying NLTK linguist resources...")
        if on_progress: on_progress("Verifying NLTK linguist resources...")
        
        try:
            logger.debug("Checking WordNet...")
            nltk.data.find('corpora/wordnet')
            
            logger.debug("Checking Averaged Perceptron Tagger...")
            nltk.data.find('taggers/averaged_perceptron_tagger')
            
            logger.debug("Checking Punkt...")
            nltk.data.find('tokenizers/punkt')
            
            logger.info("NLTK resources verified.")
            if on_progress: on_progress("NLTK resources verified.")
            
        except LookupError:
            if on_progress: on_progress("Downloading NLTK linguist data (this may take a minute)...")
            logger.info("NLTK resources missing. Starting download...")
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('punkt')
        self._initialized = True

    def generate_hard_query(self, sentence: str, on_progress: Optional[Callable[[str], None]] = None) -> str:
        self._ensure_nltk(on_progress)
        
        tokens = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)
        
        # Increase hardness: drop more common words, focus on entities
        mode = random.choices(
            ["paraphrase", "keywords", "original"], 
            weights=[0.7, 0.2, 0.1], 
            k=1
        )[0]

        if mode == "original":
            return sentence

        if mode == "keywords":
            # Keep meaningful words to ensure even hashing retrieval works
            keywords = [w for w, t in pos_tags if t.startswith(('NN', 'VB', 'JJ')) and len(w) > 2]
            if len(keywords) > 2:
                random.shuffle(keywords)
                return " ".join(keywords[:5])
            return sentence

        new_words = []
        for word, tag in pos_tags:
            if tag.startswith(('NN', 'VB', 'JJ')) and len(word) > 3:
                syns = wordnet.synsets(word)
                if syns:
                    # Get lemmas and Filter heavily
                    lemmas = {l.name().replace('_', ' ') for s in syns for l in s.lemmas()}
                    lemmas = {l for l in lemmas if l.lower() != word.lower() and "_" not in l}
                    if lemmas:
                        new_words.append(random.choice(list(lemmas)))
                        continue
            new_words.append(word)
        
        query = " ".join(new_words)
        return query

    def generate_boundary_qa(self, doc_id: str, sentences: list[str], on_progress: Optional[Callable[[str], None]] = None) -> list[dict]:
        """
        Creates QA pairs where the answer span crosses sentence boundaries.
        This tests if the chunker keeps related sentences together.
        """
        self._ensure_nltk(on_progress)
        qa = []
        for i in range(len(sentences) - 1):
            s1 = sentences[i]
            s2 = sentences[i+1]
            # Combine sentences into one answer span
            combined = s1 + " " + s2
            
            # Query is a paraphrase of the junction
            query_base = s1[-30:] + " " + s2[:30]
            query = self.generate_hard_query(query_base)
            
            qa.append({
                "id": f"{doc_id}#bqa#{i}",
                "doc_id": doc_id,
                "query": query,
                "answer_span": combined,
            })
        return qa
