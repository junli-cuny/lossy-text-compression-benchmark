# Scaled fact-level evaluation with LLM raters (paper Appendix C, Table 15)

Two LLM raters, Claude Opus 5 (`claude-opus-5`) and GPT-5.6 (`gpt-5.6-sol`), apply the protocol of the human
study (`../README.md`) to the full strategy-rate grid: 100 BBC test articles x {Step, WordLen, WordFreq, LP} x
9 rates (Gemini 2.0 Flash reconstructions from the BBC workbook; WordFreq = WordFreq-3) and 100 512-character
BBC chunks x {Entropy, Entropy-LP, Hybrid-0.5} x 5 rates (Gemini 2.5 Flash reconstructions from
`../../samples/bbc_hybrid_samples.jsonl`), 5,098 items in total.

| File | Content |
|---|---|
| `build_llm_rater_grid.py` | builds the grid: samples the articles and chunks, attaches 3-5 facts per original (the human-verified facts of the human study where the original matches, otherwise drafted with gemini-2.5-flash), shuffles, and writes `batches/`, `mapping.json`, `facts.json` |
| `batches/batch_NNN.json` | the rater inputs, blind to condition (item id, original, reconstruction, facts). `batch_000.json` holds the 54 human-study items (ids `e5_NNN` = item NNN of the human study), the calibration set |
| `mapping.json` | item id -> strategy, r_keep, decoder, source unit, and the provenance of its facts (`facts_source`) |
| `facts.json` | facts per original text (keyed by a hash of the normalised text), with their source |
| `RATER_PROMPT.md` | the system prompt and the per-item user message, identical for both raters |
| `run_llm_raters.py` | the rater runner (provider batch APIs; needs `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) |
| `annotations_claude/`, `annotations_gpt/` | per-item judgments of each rater: one label per fact with an evidence span, the invented-claim flag and its note |
| `analyze_llm_raters.py` | Table 15 (per strategy and rate: share of facts judged preserved, share of items flagged) and rater-rater agreement; writes `llm_rater_results.json` |
| `calibrate_llm_raters.py` | agreement of each rater with the four human annotators on the 54 calibration items, the raters' invented-claim flags on those items, and the human-human kappa range |

The grid was built from an export of the BBC workbook that differs from `data/BBC_news_latest.xlsx` in the
regenerated Step reconstructions at r_keep = 0.6, so `batches/` (not a rebuild) is the record of what the raters
judged. Items whose judgment failed are stored with `unresolved` set or without `facts` and are skipped by the
analysis (nine items: eight responses withheld by the Claude rater and one malformed GPT response).
