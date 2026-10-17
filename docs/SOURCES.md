# Data Sources

## 1. Europol Internet Organised Crime Threat Assessment (IOCTA) 2025

- **Report title:** Steal, Deal and Repeat: How Cybercriminals Trade and Exploit Your Data
- **Publisher:** Europol
- **Report year:** 2025
- **Official source:** https://www.europol.europa.eu/publication-events/main-reports/steal-deal-and-repeat-how-cybercriminals-trade-and-exploit-your-data
- **Local filename:** `iocta_report.pdf`
- **Local directory:** `data/reports/`
- **Usage:** Demonstration and evaluation of source-grounded information retrieval.

### Processing

The application extracts text from the PDF, divides it into passages, and records source metadata, including the document title and PDF page number. TF-IDF vectorization and cosine similarity are used to retrieve potentially relevant passages.

### Limitations

- Retrieval relevance does not guarantee factual correctness.
- Page numbers refer to PDF pages and may differ from printed page numbers.
- The current baseline retrieves evidence but does not generate a synthesized AI answer.
- The application should not be treated as an operational cybercrime investigation or intelligence system.
- Users should consult the original Europol publication to verify evidence and context.

Only use publicly available material and respect the publisher's terms when redistributing documents.
