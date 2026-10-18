"""Extractive answer generation grounded in retrieved evidence.

Selects complete sentences from retrieved report passages, cleans common
PDF extraction artifacts, avoids duplicate evidence, and preserves citations.
"""

import re

from src.rag import SearchResult


STOP_WORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that",
    "this", "from", "into", "how", "what", "why", "when", "where",
    "who", "does", "did", "can", "could", "would", "should",
    "their", "they", "them", "these", "those", "have", "has",
    "had", "been", "being", "its", "about", "also", "than",
    "then", "there", "which", "while", "create", "creates",
    "created", "risk", "risks", "do", "is", "it", "of", "to",
    "in", "on", "by", "as", "at", "an", "a", "or", "be", "not",
    "may", "might", "use", "uses", "used", "using", "their",
}


FOOTNOTE_MARKERS = {"A", "B", "C"}


TECHNIQUE_TERMS = {
    "exploit", "exploits", "exploitation", "kits", "brute",
    "force", "supply", "chain", "social", "engineering",
    "phishing", "vulnerabilities", "credentials", "attacks",
    "attack", "compromise", "misconfigured", "malware",
    "malicious", "attachments", "urls", "websites", "dropper",
    "botnet", "malvertising", "infostealers", "infostealer",
    "distribution", "distribute", "campaigns", "advertising",
}


def _words(text: str) -> set[str]:
    """Extract meaningful lowercase terms."""
    return {
        word.lower()
        for word in re.findall(r"[a-zA-Z0-9]+", text)
        if len(word) > 2 and word.lower() not in STOP_WORDS
    }


def _clean_text(text: str) -> str:
    """Clean common PDF extraction artifacts without inventing content."""
    text = text.replace("\u00a0", " ")

    # Normalize typographic characters.
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\ufb01": "fi",
        "\ufb02": "fl",
    }

    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    # Repair known words split during extraction.
    repairs = [
        (r"\bHybrid\s+thr\s+eat\b", "hybrid threat"),
        (r"\bEurop\s+ol\b", "Europol"),
        (r"\bthr\s+eat\b", "threat"),
        (r"\bAdv\s+anced\b", "Advanced"),
        (r"\bz\s+ero-day\b", "zero-day"),
        (r"\bcyber\s+criminals\b", "cybercriminals"),
        (r"\bsupply\s*-\s*chain\b", "supply-chain"),
    ]

    for pattern, replacement in repairs:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Remove URLs.
    text = re.sub(r"https?://\S+", " ", text)

    # Remove known report title/header text.
    text = re.sub(
        r"STEAL,\s*DEAL\s+AND\s+REPEAT\s*-\s*"
        r"HOW\s+CYBERCRIMINALS\s+TRADE\s+AND\s+EXPLOIT\s+YOUR\s+DATA",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bIOCTA\s+2025\b", " ", text, flags=re.IGNORECASE)

    # Repair malformed APT footnote labels without deleting later evidence.
    text = re.sub(
        r"\bA\s+Advanced\s+persistent\s+threat\s*\(APT\)",
        "Advanced persistent threat (APT)",
        text,
        flags=re.IGNORECASE,
    )

    # Remove attached numeric superscript-style footnote references:
    # for example, "actors12" becomes "actors".
    text = re.sub(r"(?<=[A-Za-z])\d{1,2}\b", "", text)

    # Remove footnote letters attached to lowercase words:
    # "actorsB" becomes "actors".
    text = re.sub(r"(?<=[a-z])[ABCF]\b", "", text)

    # Remove footnote letters following a closing parenthesis:
    # "(APT)A" becomes "(APT)".
    text = re.sub(r"(?<=\))[ABCF]\b", "", text)

    # Remove standalone footnote letters after punctuation.
    text = re.sub(r"(?<=[.!?])\s+[ABC]\b", "", text)

    # Remove page numbers appearing at the beginning of extracted text.
    text = re.sub(r"^\s*\d{1,3}\s+(?=[A-Z])", "", text)

    return " ".join(text.split())


def _split_sentences(text: str) -> list[str]:
    """Split normalized text into candidate sentences."""
    # PDF extraction may lowercase the first word of a new sentence.
    # Split after sentence punctuation even when the next word is lowercase.
    parts = re.split(r"(?<=[.!?])\s+", text)

    return [
        part.strip(" \t\r\n-")
        for part in parts
        if part.strip(" \t\r\n-")
    ]


def _complete_sentence(sentence: str) -> bool:
    """Reject incomplete fragments, headings, and publication boilerplate."""
    sentence = sentence.strip()

    if not sentence:
        return False

    # Reject passages that visibly begin with a page number.
    if re.match(r"^\d+\s+", sentence):
        return False

    # Reject common fragments beginning with a closing abbreviation marker.
    if re.match(r"^(?:APTs?|IABs?)\)\s+", sentence, flags=re.IGNORECASE):
        return False

    # Reject sentences with too little substance.
    if len(sentence.split()) < 8:
        return False

    # Require terminal sentence punctuation.
    if not re.search(r'[.!?]["\')\]]?$', sentence):
        return False

    words = re.findall(r"[a-zA-Z]+", sentence)

    if len(words) < 6:
        return False

    # Reject lowercase starts, which commonly indicate a mid-sentence fragment.
    if not sentence[0].isupper():
        return False

    # Reject incomplete endings.
    if words[-1].lower() in {
        "and", "or", "but", "while", "because", "including",
        "such", "which", "that", "to", "of", "for", "with",
        "by", "as", "from", "on", "in", "at",
    }:
        return False

    rejected_phrases = (
        "accessible at https",
        "hybrid threat actors can be state or non-state actors",
        "considering europol",
        "cite this publication",
        "how is data exploited",
        "how are data and access acquired",
        "steal, deal and repeat",
    )

    lowered = sentence.casefold()

    if any(phrase in lowered for phrase in rejected_phrases):
        return False

    return True


def _question_focus(question: str) -> set[str]:
    """Extract meaningful terms from the question."""
    return _words(question)


def _expand_question_terms(question_words: set[str]) -> set[str]:
    """Expand common cybersecurity topics to improve sentence matching."""
    expanded = set(question_words)

    if question_words.intersection({
        "technique", "techniques", "method", "methods",
    }):
        expanded.update(TECHNIQUE_TERMS)

    if question_words.intersection({
        "advanced", "threat", "actor", "actors", "apt",
    }):
        expanded.update({
            "advanced", "threat", "actors", "hybrid",
            "apt", "sophisticated", "cybercriminals",
        })

    if question_words.intersection({
        "data", "steal", "stolen", "exploit", "exploitation",
    }):
        expanded.update({
            "data", "personal", "stolen", "credentials",
            "access", "information", "trade", "exploit",
        })

    if question_words.intersection({
        "pose", "poses", "threat", "threats", "risk", "risks",
    }):
        expanded.update({
            "espionage", "coercion", "disruption",
            "instability", "governments", "infrastructure",
            "theft", "destruction", "malicious", "activity",
        })

    return expanded


def _is_reference_sentence(sentence: str) -> bool:
    """Reject bibliographic references and citation-only sentences."""
    lowered = sentence.casefold()

    signals = (
        "available at http",
        "retrieved from http",
        "doi.org/",
        "all rights reserved",
        "isbn ",
    )

    if any(signal in lowered for signal in signals):
        return True

    # A reference-like author/year opening is not substantive evidence.
    if re.match(
        r"^[A-Z][A-Za-z&.' -]{1,60},\s*(?:19|20)\d{2}\b",
        sentence.strip(),
    ):
        return True

    return False


def generate_answer(question: str, results: list[SearchResult]) -> str:
    """Select relevant, complete evidence sentences with page citations."""
    if not question or not question.strip() or not results:
        return (
            "I could not find sufficiently relevant evidence in the "
            "currently indexed documents. No factual answer is provided."
        )

    question_words = _question_focus(question)

    if not question_words:
        return (
            "The question does not contain enough meaningful terms "
            "to identify relevant evidence."
        )

    expanded_terms = _expand_question_terms(question_words)
    candidates = []

    for result in results:
        chunk = result.chunk
        cleaned_text = _clean_text(chunk.text)

        for raw_sentence in _split_sentences(cleaned_text):
            # Clean each sentence independently before evaluating it.
            sentence = _clean_text(raw_sentence)

            if not _complete_sentence(sentence):
                continue

            if _is_reference_sentence(sentence):
                continue

            sentence_words = _words(sentence)
            overlap = expanded_terms.intersection(sentence_words)

            if not overlap:
                continue

            # Measure coverage against the original question terms.
            # Expanded terms help matching but should not penalise
            # sentences for missing every possible related concept.
            original_overlap = sentence_words & question_words
            coverage = len(original_overlap) / max(len(question_words), 1)
            precision = len(overlap) / max(len(sentence_words), 1)

            # Combine retrieval relevance with question-term overlap.
            score = (
                0.30 * result.score
                + 0.45 * coverage
                + 0.25 * precision
            )

            # Prioritise concrete delivery methods for malware questions.
            is_malware_delivery_question = (
                "malware" in question_words
                and "delivery" in question_words
            )
            delivery_terms = {
                "phishing", "attachments", "urls", "websites",
                "malvertising", "dropper", "botnet", "malware",
                "infostealers", "infostealer", "distribution",
                "distribute", "advertising",
            }

            if question_words.intersection({
                "technique", "techniques", "method", "methods",
            }):
                if sentence_words.intersection(TECHNIQUE_TERMS):
                    score += 0.10

            if is_malware_delivery_question:
                if sentence_words.intersection(delivery_terms):
                    score += 0.30
                if (
                    "personal" in sentence_words
                    and "access" in sentence_words
                    and not sentence_words.intersection(delivery_terms)
                ):
                    score -= 0.15

            candidates.append({
                "sentence": sentence,
                "score": score,
                "page": chunk.page,
                "title": chunk.title,
                "url": chunk.url,
                "source": chunk.source,
                "words": sentence_words,
            })

    if not candidates:
        return (
            "Relevant passages were retrieved, but no complete sentence "
            "with matching terms could be identified. "
            "No factual answer is provided."
        )

    # Avoid presenting weak keyword matches as factual answers.
    candidates = [
        item for item in candidates
        if item["score"] >= 0.08
    ]

    if not candidates:
        return (
            "Relevant passages were retrieved, but the available sentences "
            "did not meet the answer relevance threshold. "
            "No factual answer is provided."
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)

    selected = []
    seen_sentences = set()

    for candidate in candidates:
        normalized = re.sub(
            r"\W+", " ", candidate["sentence"].casefold()
        ).strip()

        if normalized in seen_sentences:
            continue

        # Avoid near-duplicate sentences using Jaccard similarity.
        too_similar = False

        for previous in selected:
            union = candidate["words"] | previous["words"]
            intersection = candidate["words"] & previous["words"]

            similarity = len(intersection) / max(len(union), 1)

            if similarity > 0.80:
                too_similar = True
                break

        if too_similar:
            continue

        selected.append(candidate)
        seen_sentences.add(normalized)

        if len(selected) == 3:
            break

    if not selected:
        return "No sufficiently reliable sentence could be selected."

    lines = ["Based on the retrieved evidence:"]

    for number, item in enumerate(selected, start=1):
        lines.append(
            f"- {item['sentence']} [{number}, p. {item['page']}]"
        )

    lines.append(
        "\nThis is an extractive answer: sentences are selected from "
        "retrieved passages, not generated by an LLM. Check the cited "
        "pages in the original report before relying on individual claims."
    )

    lines.append("\nSources:")

    for number, item in enumerate(selected, start=1):
        source_line = (
            f"[{number}] {item['title']} - {item['source']}, "
            f"page {item['page']}"
        )

        if item["url"]:
            source_line += f" - {item['url']}"

        lines.append(source_line)

    return "\n".join(lines)
