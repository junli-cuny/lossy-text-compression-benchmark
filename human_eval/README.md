# Human fact-level study (paper Appendix B, Table 14)

54 BBC News reconstructions, {WordFreq-6, LP} x r_keep in {0.5, 0.3, 0.1}, 9 per cell, judged blind to
strategy and rate by four annotators (`annotator1`-`annotator4`). For each item the annotator verifies 3-5
atomic facts taken from the original, labels each fact in the reconstruction as preserved (P), distorted (D)
or missing (M), and flags whether the reconstruction asserts an invented claim. The file prefix `e5_` is the
study's internal code.

| File | Content |
|---|---|
| `build_e5_annotation_sheet.py` | how the 54 items were sampled from the BBC workbook (seeded); writes `e5_annotation_sheet_rebuilt.xlsx` so the shipped sheet is not overwritten |
| `e5_annotation_sheet.xlsx` | the item sheet; sheet `key` maps `item_id` to strategy (`wordfreq6` = WordFreq-6, `opt_prepolish` = LP), r_keep and the workbook row (`chunk_id`) |
| `e5_items.json` | the 54 items (original, reconstruction) |
| `draft_e5_facts.py` | drafts 3-5 facts per item from the original only, with gemini-2.5-flash (API key needed); writes `e5_facts_drafted.json` |
| `e5_facts.json` | the verified facts the annotators were shown; the tool let an annotator correct a fact against the original before judging it, and each annotation file records the fact text that was judged |
| `e5_annotator_input.json`, `E5_Annotation_Tool.html`, `e5_annotation_ui_shell.html` | the annotation tool (instructions included) with its item data, and its empty template |
| `e5_annotations_annotator{1,2,3,4}.json` | the four annotators' judgments (1,062 fact judgments in total) |
| `run_e5_analysis.py` | Table 14 (`--table14`), pairwise Cohen's kappa (three-way and binary), invented-claim flags, and the correlation of fact preservation with BERTScore; writes `e5_analysis.json` |
| `e5_halluc_adjudicated.json` | the four items adjudicated as asserting an invented claim |
| `llm_raters/` | the scaled LLM-rater evaluation of Appendix C (Table 15), calibrated on these 54 items |

`run_e5_analysis.py` reads the BERTScore and ROUGE-L of the same reconstructions from
`../results/e4_per_chunk.csv.gz` (distilroberta-base BERTScore).
