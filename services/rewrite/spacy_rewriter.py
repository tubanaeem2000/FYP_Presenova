"""
spaCy Rule-Based Slide Rewriter Engine (Local / Deterministic)

Performs text enhancement, passive-to-active voice conversion,
conciseness filler phrase substitution, and 6x6 bullet splitting.
"""

import logging
import re
import sys
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

_SPACY_NLP = None


def _load_spacy():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            import spacy
            _SPACY_NLP = spacy.load("en_core_web_sm")
            logger.info("[spacy_rewriter] Loaded spaCy model en_core_web_sm")
        except Exception as exc:
            logger.warning("[spacy_rewriter] Could not load spaCy model: %s", exc)
            _SPACY_NLP = False
    return _SPACY_NLP if _SPACY_NLP is not False else None


# Filler Phrase Lookup Dictionary
FILLER_SUBSTITUTIONS = {
    r"\bin order to\b": "to",
    r"\bdue to the fact that\b": "because",
    r"\bat this point in time\b": "now",
    r"\ba large number of\b": "many",
    r"\btake into consideration\b": "consider",
    r"\buntil such time as\b": "until",
    r"\bfor the purpose of\b": "for",
    r"\bin the event that\b": "if",
    r"\bwith reference to\b": "about",
    r"\bhas the capability to\b": "can",
    r"\bis of the opinion that\b": "believes",
    r"\bmake a decision\b": "decide",
}


def substitute_filler_phrases(text: str) -> str:
    """Substitute verbose/filler phrases with concise alternatives."""
    result = text
    for pattern, replacement in FILLER_SUBSTITUTIONS.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def convert_passive_to_active(sentence: str) -> str:
    """Detect passive voice (nsubjpass) via spaCy and restructure to active voice."""
    nlp = _load_spacy()
    if not nlp or len(sentence.split()) < 4:
        return sentence

    try:
        doc = nlp(sentence)
        nsubjpass = None
        verb = None
        agent = None

        for token in doc:
            if token.dep_ == "nsubjpass":
                nsubjpass = token.text
            elif token.dep_ == "auxpass" and token.head.pos_ == "VERB":
                verb = token.head.lemma_
            elif token.dep_ == "agent":
                agent_objs = [child.text for child in token.children if child.dep_ == "pobj"]
                if agent_objs:
                    agent = agent_objs[0]

        if nsubjpass and verb and agent:
            # Simple active reconstruction: "[Agent] [Active Verb] [Passive Subject]"
            active = f"{agent.capitalize()} {verb}s {nsubjpass.lower()}."
            return active
    except Exception as exc:
        logger.warning("[spacy_rewriter] Passive-to-active conversion error: %s", exc)

    return sentence


def split_bullet_point(bullet: str, max_words: int = 15) -> List[str]:
    """Split bullet points exceeding max_words on conjunctions or semicolons."""
    words = bullet.split()
    if len(words) <= max_words:
        return [bullet]

    # Split on semicolon or conjunctions
    parts = re.split(r';|\b(?:and|but|however|whereas|furthermore)\b', bullet, flags=re.IGNORECASE)
    cleaned_parts = [p.strip().capitalize() for p in parts if len(p.strip()) > 5]

    if len(cleaned_parts) > 1:
        return cleaned_parts
    
    # Fallback to word slicing if no punctuation boundary
    half = len(words) // 2
    part1 = " ".join(words[:half]).strip().capitalize()
    part2 = " ".join(words[half:]).strip().capitalize()
    return [part1, part2]


def rewrite_slide_content(slide_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rewrite a single slide's title and bullet points using spaCy rules.
    Enforces 6x6 rule (max 6 bullets, max 15 words per bullet).
    """
    title = slide_data.get("title", "")
    bullets = slide_data.get("bullet_points", [])

    # Clean title
    clean_title = substitute_filler_phrases(title).title()

    new_bullets = []
    for b in bullets:
        # Step 1: Substitute filler phrases
        concise = substitute_filler_phrases(b)

        # Step 2: Passive to active
        active = convert_passive_to_active(concise)

        # Step 3: Enforce 15-word bullet limit
        split_bullets = split_bullet_point(active, max_words=15)
        new_bullets.extend(split_bullets)

    # Step 4: Enforce max 6 bullets per slide (6x6 rule)
    if len(new_bullets) > 6:
        new_bullets = new_bullets[:6]

    return {
        "slide_number": slide_data.get("slide_number", 1),
        "title": clean_title,
        "bullet_points": new_bullets,
        "speaker_notes": slide_data.get("speaker_notes", ""),
    }
