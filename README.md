# live-data-comparison

Two versions of the same agent - one that can only see my own indexed research papers,
one that can also search the live web. Ran both on 10 questions to see where fresh data
actually helps vs where it just adds risk. Runs on a local LLM (Llama 3.2 3B via LM
Studio) + free Tavily search, so this cost nothing to build.

## Corpus

Static agent reads from a chroma index of ~13 Sinhala NLP papers I already had indexed
(2018 MT thesis, a 2014 NER paper, a 2019 survey on Sinhala NLP resources, etc). Wrote
the questions to match what's actually in there instead of using placeholder topics.

## Files

- `static_agent.py` - embeds the question, pulls top-3 chunks from chroma, answers from that only
- `live_search_tool.py` - Tavily web search wrapper, with retries since it dropped connections a couple times mid-run
- `full_agent.py` - decide-loop agent, picks search_papers or web_search each round (up to 3), then answers from whatever it gathered
- `run_comparison.py` - runs both agents on all 10 questions, saves after every single one so a crash doesn't wipe the run
- `questions.py` - 5 questions answerable from the papers, 5 that need current 2025/2026 info
- `ground_truth.json` - my own researched answers to score against

## Bugs I ran into (more useful than the final number honestly)

**Context window overflow.** LM Studio had the model loaded with a 2048 token limit.
Once I started tagging retrieved chunks by source, a few questions blew past that and
crashed with a 400 error. Bumped the context length in LM Studio and capped each chunk
to ~800 chars as a backup.

**Model ignoring its own retrieved context.** Asked who wrote the thesis - retrieval
correctly pulled just 2 names (Ranathunga, Tennage), but the model's answer added 3 more
names anyway, from a completely different (real) paper with a similar title that it
apparently knew from training. Confirmed this with a debug print - correct 2-name
context went in, 5 names came out. Small local models just don't reliably stick to
"only use the provided context" even when you tell them to.

**Overcorrected the fix.** Tried fixing the above with an explicit instruction ("if
context lists 2 authors, answer with exactly 2"). Fixed that one question, broke
everything else - model started refusing or dumping random names on questions that had
nothing to do with authors at all. Made the instruction generic instead, which fixed the
refusals but the original author hallucination came back on that specific question.
Decided to stop tuning here instead of chasing it forever - this is a real limitation of
a 3B model holding two competing instructions at once, not something one more sentence
was going to fix.

**Tavily dropping connections.** Added retry logic (3 attempts, short delay) and made
run_comparison.py save after every question instead of only at the end. Both actually
fired for real on the next run and saved it from crashing out.

## Results

| # | Question | Static-only | Live-search |
|---|---|---|---|
| 1 | Language pair of the thesis | Couldn't state it | Couldn't state it |
| 2 | Google Translate effectiveness | Vague | Honest "not enough info" |
| 3 | NER method + year | Got the year right, no method | Wrong - pulled a different NER paper |
| 4 | Why Sinhala is resource-poor | Correct, real stats | Correct |
| 5 | Thesis authors | Right names but hedges weirdly | Wrong - hallucinated 3 extra names |
| 6 | 2026 low-resource LLM approaches | Honest, dated | Partial, missed the obvious example |
| 7 | New Sinhala LLM released | Honest "don't know" | Correctly named SinLlama |
| 8 | Sinhala-Tamil MT SOTA 2026 | Honest, dated | Partial, real but old citations |
| 9 | Recent Sinhala datasets/tools | Honest, unclear | Reasonable grounded answer |
| 10 | 2026 NER techniques generally | Honest, general | Honest, flagged its own sources as old |

## The actual takeaway

Live search only clearly won on the one question it was built for (#7, something
genuinely time-sensitive). Everywhere else it either matched the static agent or
introduced a new failure the static agent didn't have (#5). Adding web search to an
agent helps exactly where you'd expect and adds real risk everywhere else - especially
on a small local model that doesn't always keep its own instructions straight.

## Running it

Needs LM Studio running with `llama-3.2-3b-instruct` loaded (context length >= 4096) and
a free Tavily key.

```bash
pip install openai tavily-python sentence-transformers chromadb
set TAVILY_API_KEY=your-key-here

python static_agent.py
python full_agent.py
python run_comparison.py
```

`run_comparison.py` resumes automatically if it gets interrupted partway through.
