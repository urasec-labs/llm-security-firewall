A modular, high-performance security firewall designed to inspect, detect, and mitigate threats in Large Language Model (LLM) applications.

This project focuses on protecting LLM-based systems against vulnerabilities outlined in the **OWASP Top 10 for LLM Applications**, including Prompt Injections, Data Leakage, Sensitive Information Disclosure (PII), and Unauthorized Actions.

---

## 🏗️ Architecture Overview

The framework processes incoming user inputs and outgoing model responses through a multi-layered guard mechanism:

[User Input]
│
▼
┌──────────────────────────────────────────┐
│              Input Guard                 │
│  ├─ Rule-Based Engine (Regex/Patterns)   │
│  ├─ Semantic Threat Detection            │
│  └─ ML Classifier (Prompt Injection)     │
└────────────────────┬─────────────────────┘
│
▼
[Upstream LLM Engine]
│
▼
┌──────────────────────────────────────────┐
│              Output Guard                │
│  ├─ PII / Sensitive Data Masking         │
│  └─ Toxicity & Hallucination Filter      │
└────────────────────┬─────────────────────┘
│
▼
[Safe Response]


---

## ✨ Key Features

* **Input & Output Guards:** Full control over prompts and generated outputs.
* **Multi-Layer Detectors:**
  * `rules.py`: Fast regex-based signature matching for known attack payloads.
  * `semantic.py`: Context-aware threat analysis using semantic evaluation.
  * `classifier.py`: ML-based classification for prompt injection and jailbreak detection.
* **Full-Stack Ecosystem:** Includes a Node.js/Express API gateway and a React-based mockup sandbox for real-time testing.
* **Audit & Access Control:** Detailed query logging and RBAC middleware support.

---

## 🚀 Quick Start

### Prerequisites
* Python 3.10+
* Node.js 18+ and `pnpm`

### Setup & Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/urasec-labs/llm-security-firewall.git](https://github.com/urasec-labs/llm-security-firewall.git)
   cd llm-security-firewall
Install dependencies:

Bash
pnpm install
Start the API Server & Sandbox:

Bash
pnpm dev
🔬 Research Goals & Roadmap
[ ] Integration of fine-tuned Transformer models (e.g., RoBERTa) specialized for prompt injection classification.

[ ] Real-time security auditing for Retrieval-Augmented Generation (RAG) pipelines.

[ ] Benchmarking firewall latency impact on high-throughput LLM endpoints.

👤 Author & Lab
Developed and maintained by Uras Akas under urasec-labs.

Portfolio: urasakas.com

LinkedIn: linkedin.com/in/uras-akas


---

