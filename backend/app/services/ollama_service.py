"""Ollama LLM client for query expansion and result explanation."""

from __future__ import annotations

import httpx


_EXPAND_PROMPT = """\
The user wants to: "{query}"

The user wants to accomplish the following task: "{query}"

Your job is to rewrite this request into a short retrieval-friendly query that better matches the terminology and skill descriptions likely stored in the database. 

Identify the most relevant technical skills, tools, frameworks, and implementation areas related to the user's goal. Use the available skill categories below as a guide, and include only the categories that are truly relevant:

- Frontend frameworks: React, Vue, Angular, Next.js, Svelte, Astro
- UI/design: Tailwind, shadcn-ui, accessibility, Figma
- Backend: Node.js, Python, FastAPI, Express, GraphQL, REST APIs
- Databases: PostgreSQL, MongoDB, Prisma, Redis, Supabase
- DevOps/cloud: Docker, AWS, Cloudflare, CI/CD, Terraform, Containers, Kubernetes
- Testing: Vitest, Playwright, Jest, pytest
- AI/agents: LLM integration, MCP tools, RAG, embeddings, Claude/OpenAI SDKs, OpenClaw
- Mobile: Swift, iOS, Android, Flutter
- Security: auth, OAuth, JWT
- Developer productivity: documentation, markdown, code quality
- Productivity: Powerpoints, Notion, Slack, Miro, Excel

Write a concise expanded query in 2-4 sentences that:
1. Restates the user's goal in clearer technical terms
2. Adds the most relevant skill areas, tools, and frameworks
3. Uses wording that would help match existing database entries
4. Avoids adding irrelevant categories or unnecessary detail

Return ONLY the expanded query text, with no bullet points, labels, or explanation.
"""

_EXPLAIN_PROMPT = """\
The user wants to: "{query}"

Recommended skills:
{skills}

For each skill write ONE line in this exact format:
[skill-name] — [one sentence how it specifically helps the user accomplish their goal]

If two skills are very similar in purpose, group them on one line:
[skill-a] / [skill-b] — [one sentence, note which is more relevant]

Output ONLY the lines, one per skill or group. No intro, no summary, no extra text.\
"""


class OllamaService:
    """Calls a local Ollama instance for query expansion and result explanation."""

    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _generate(self, prompt: str) -> str:
        """Call /api/generate and return the response text. Raises on failure."""
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def expand_query(self, query: str) -> str:
        """Return a skill-focused expanded version of the user query.

        Falls back to the original query on any error (Ollama not running, timeout, etc.).
        """
        try:
            prompt = _EXPAND_PROMPT.format(query=query)
            expanded = self._generate(prompt)
            return expanded if expanded else query
        except Exception:
            return query

    def explain_results(self, query: str, skills: list[tuple[str, str]]) -> list[str]:
        """Return per-skill explanation lines for why the skills fit the user's goal.

        Each line is either:
          "skill-name — one sentence why it helps"
          "skill-a / skill-b — grouped if similar"

        Returns an empty list on any error.
        """
        try:
            skills_text = "\n".join(
                f"{i + 1}. {name} — {desc[:100]}"
                for i, (name, desc) in enumerate(skills)
            )
            prompt = _EXPLAIN_PROMPT.format(query=query, skills=skills_text)
            raw = self._generate(prompt)
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            return lines
        except Exception:
            return []
