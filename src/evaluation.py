"""Small, manually page-labelled retrieval evaluation set."""

QUESTIONS = [
    {
        "question": "How do criminals steal and trade personal data?",
        "expected_pages": {9, 12},
    },
    {
        "question": "How do cybercriminals use stolen credentials?",
        "expected_pages": {12, 13},
    },
    {
        "question": "What role do infostealers play in cybercrime?",
        "expected_pages": {13},
    },
    {
        "question": (
            "How do cybercriminals distribute malware through phishing, "
            "infostealers, malicious websites and droppers?"
        ),
        "expected_pages": {13, 14, 17},
    },
    {
        "question": "How do criminals monetize stolen data?",
        "expected_pages": {9, 17},
    },
]
