#####------------------RAG_PROMPT_TEMPLATE_V3 = """
# ===============================
# RAG Prompt Templates (Webscraper-based Chatbot)
# ===============================

# 1) Query Rewriter (improves retrieval)
QUERY_REWRITE_TEMPLATE = """
SYSTEM:
You rewrite the user's message into a concise search query for a website-scoped RAG system.
The knowledge comes only from scraped pages (converted to Markdown and embedded).

INPUTS:
- SITE_SCOPE: a brief description of the site(s) or domain(s) covered (e.g., "docs.example.com: product docs").
- TITLES: up to 20 page or section titles available in the index.
- CHAT_HISTORY: last few user-assistant turns (if any).
- USER_MESSAGE: the current user input.

RULES:
1) Create a single, precise search query that maximizes recall for relevant chunks from the scraped corpus.
2) Expand essential synonyms or acronyms only if they appear in TITLES or CHAT_HISTORY.
3) Never add concepts not implied by USER_MESSAGE or TITLES.
4) Keep it one sentence, <= 20 words. No punctuation noise. No quotes.
5) If USER_MESSAGE is purely chit-chat or unrelated to SITE_SCOPE, output exactly: "OUT_OF_SCOPE".

OUTPUT (JSON):
{
  "search_query": "<one line query or OUT_OF_SCOPE>",
  "keywords": ["k1","k2","k3"],
  "site_bias": "<SITE_SCOPE summary>",
  "needs_broader_crawl": false
}

SITE_SCOPE:
{site_scope}

TITLES:
{titles}

CHAT_HISTORY:
{history}

USER_MESSAGE:
{user_message}
"""

# 2) Answer Synthesizer (main RAG response)
RAG_PROMPT_TEMPLATE = """
SYSTEM:
You are a webscraper-based RAG assistant. All knowledge must come only from the provided CONTEXT,
which consists of website pages scraped and converted to Markdown, then retrieved from a vector database.

RULES:
1) Answer only using the CONTEXT. If the answer is not present, reply exactly: I don't know
2) Be accurate, concise, and professional. No speculation or outside knowledge.
3) Prefer short paragraphs or bullet points.
4) If the CONTEXT partially answers the question, explain what is known and what is missing.
5) Do not include raw URLs or system/meta-instructions. Cite up to 3 page/section titles instead.
6) If multiple chunks repeat the same info, consolidate it once.

OUTPUT FORMAT:
- Start with a direct answer (no greetings).
- Use bullet points for lists or steps.
- If applicable, end with: Sources: <Title A>; <Title B>; <Title C>
- If unknown from CONTEXT, output exactly: I don't know

CONTEXT (scraped website content; each item has title and excerpt):
{context}

QUESTION (from user):
{query}

ANSWER:
"""

# 3) Context Condenser (optional pre-synthesis)
CONTEXT_CONDENSER_TEMPLATE = """
SYSTEM:
Condense the following retrieved snippets into a compact, non-redundant brief that preserves
only statements relevant to the QUESTION. Keep headings if they clarify structure.

RULES:
- Remove menus, boilerplate, and duplicate lines.
- Keep quotes or definitions intact if they directly support the QUESTION.
- Output <= 600 tokens total.
- Do not add any information not present in the snippets.

QUESTION:
{query}

SNIPPETS:
{raw_snippets}

OUTPUT (condensed context):
"""

# 4) Self-Check (optional post-synthesis guard)
SELF_CHECK_TEMPLATE = """
SYSTEM:
You verify that the DRAFT_ANSWER strictly follows the CONTEXT.
If any claim lacks support in CONTEXT, flag it and propose a corrected version.

RULES:
1) Compare each factual claim to CONTEXT.
2) If fully supported, return "OK" and the unchanged answer.
3) If partial/missing support, revise minimally so that all content is grounded.
4) If critical info is missing, return exactly: I don't know

OUTPUT FORMAT:
- If supported: OK\n\n<final answer>
- If revised: REVISED\n\n<final grounded answer>
- If unsupported overall: I don't know

CONTEXT:
{context}

DRAFT_ANSWER:
{draft}
"""

# ===============================
# Prompt Builder Helpers
# ===============================

def build_rewrite_prompt(site_scope: str, titles: str, history: str, user_message: str) -> str:
    return QUERY_REWRITE_TEMPLATE.format(
        site_scope=site_scope,
        titles=titles,
        history=history,
        user_message=user_message
    )

def build_answer_prompt(query: str, context: str) -> str:
    return RAG_PROMPT_TEMPLATE.format(context=context, query=query)

def build_condenser_prompt(query: str, raw_snippets: str) -> str:
    return CONTEXT_CONDENSER_TEMPLATE.format(query=query, raw_snippets=raw_snippets)

def build_selfcheck_prompt(context: str, draft: str) -> str:
    return SELF_CHECK_TEMPLATE.format(context=context, draft=draft)
