# Text-Preserving Lossy Text Compression

Code, data, and per-sample records for the paper
*Text-Preserving Lossy Text Compression: A Study of Strategic Deletion and LLM Reconstruction*
(Yuchun Zou, Junhong Tong, Jun Li; CUNY), AACL-IJCNLP 2026: the 5th Conference of the Asia-Pacific Chapter of the
Association for Computational Linguistics and the 15th International Joint Conference on Natural Language Processing.
Repository: https://github.com/junli-cuny/lossy-text-compression-benchmark

An encoder deletes characters or words from a text under a retention budget r_keep; the skeleton stays text; an
LLM decoder reconstructs the original from the skeleton alone. The paper is in `paper/`.

## Layout

- `data/` - the source workbooks: texts, skeletons, reconstructions and stored scores (see "Data" below).
- `notebooks/` - the Google Colab pipeline of the word-level strategies (Step, WordLen, WordFreq-3/-6, LP), their
  Gemini reconstruction and scoring, and the QLoRA fine-tuning, as run for the paper. Each notebook opens with a
  cell that maps its section names to the paper's.
- `harness/` - scripts for the semantic and hybrid strategies (Entropy, Entropy-LP, Entropy-in-FreqBuckets,
  Hybrid-alpha) with per-sample persistence and resume, the skeleton-side entity analysis, the ROUGE-L of the BBC
  word-level tables, the WordFreq-3 deletion rule as a module, and a stand-alone downstream-task tool. The
  `*_results.json` files here hold only the bucket calibration of the published runs, which a re-run reuses.
- `analysis/` - the scripts behind the remaining tables: re-runs on one decoder, scoring, entity metrics, lexical
  metrics, levers, storage accounting, LLMLingua-2, downstream tasks, r_keep = 0.05, the decoder version check and
  the LP objective comparison.
- `samples/` - per-sample records (reference, skeleton, reconstruction, decoder model), one JSON line per
  (method, retention rate, sample).
- `results/` - aggregate result files behind the tables.
- `human_eval/` - the blind human fact-level study (Appendix B) and, in `human_eval/llm_raters/`, the scaled
  evaluation with two LLM raters (Appendix C). See the README files there.
- `sft/` - configs and train/validation/test splits of every QLoRA adapter behind a Llama number in the paper, a
  re-evaluation script, and its per-sample outputs. See `sft/README.md`.
- `figures/` - the script and PDF of Figure 1.
- `paper/` - the paper (PDF) and, in `paper/latex/`, its LaTeX source: `main.tex`, `references.bib`, the
  official ACL style files and the Figure 1 PDF. Build with `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Reproducing the paper's tables

Scripts marked (API) call Gemini; everything else runs offline from the shipped files. Run scripts from the
repository root, e.g. `python analysis/score_table3.py`; paths are resolved relative to each script.

| Paper | What | Where |
|---|---|---|
| Table 1 | character-level rules (Step, Gaussian, Bernoulli, Poisson), 100 BBC chunks, Gemini 2.5 Flash | `analysis/run_table1_charlevel.py` (API; Step rule in `analysis/step_deletion.py`) -> `samples/table1_charlevel_samples.jsonl`; `analysis/score_table1.py` -> `results/table1_scores.json` |
| Table 2 | word-level strategies: BERTScore, ROUGE-L, entity recall | BERTScore: `data/BBC_news_latest.xlsx`, sheets `adaptive_word_length_deletion_s` (WordLen), `wf_3classes_stats_result` (WordFreq), `wf_optimal_stats_result` (LP), row `F1_chunk_mean`; ROUGE-L: `harness/compute_bbc_rougel.py`; entity recall: `analysis/run_e1_ner.py` -> `results/e1_ner_results.json` (source `bbc`) |
| Table 3 | all methods on the same 100 BBC chunks, Gemini 2.5 Flash | samples from `analysis/run_table3_wordfreq_summ.py` (API; WordFreq-3, Summ.), `analysis/run_e0_bbc_hybrid.py` (API; Entropy, Entropy-LP, Hybrid-0.5) and `analysis/run_e3_llmlingua.py` + `analysis/run_e3_llmlingua_extra_rates.py` (API; LLMLingua-2); every column scored by `analysis/score_table3.py` -> `results/table3_scores.json` |
| Table 4 | encoder latency | not scripted (wall-clock timing of the encoders; the hardware is not recorded) |
| Table 5 | entity recall and unsupported-entity rate after reconstruction | word-level rows: `analysis/run_e1_ner.py` -> `results/e1_ner_results.json`; semantic rows: `analysis/run_ext_merge.py` -> `results/e1_ner_results_ext.json` |
| Table 6 | QLoRA Llama vs zero-shot Llama vs Gemini | `data/BBC_news_latest.xlsx`, sheets `sft_generation_stats_result` (SFT) and `base_sft_stats_result` (zero-shot Llama), row `F1_chunk`; the Gemini column repeats Table 2. Pipeline: `notebooks/bbc_word_level_pipeline_v2.ipynb` (Way 4, Way 4.1); adapters: `sft/` |
| Table 7 | WordFreq-3 per-rate mean, std, 95% CI | `data/BBC_news_latest.xlsx`, sheet `wf_3classes_stats_result`, rows `F1_chunk_mean` / `F1_chunk_std` (CI = mean +- 1.96 std / sqrt(200)) |
| Table 8 | CER of the word-level strategies | row `cer_mean` of the sheets `benchmark_stats_result` (Step), `adaptive_word_length_deletion_s`, `wf_3classes_stats_result`, `wf_optimal_stats_result`; the Step cell at r_keep = 0.6 comes from `analysis/rescore_step_r06.py` (the stored cell predates the regenerated Step outputs) |
| Table 9 | ROUGE-L of the word-level strategies | `harness/compute_bbc_rougel.py` |
| Table 10 | ROUGE-L of the semantic and hybrid strategies | `analysis/run_ext_merge.py` -> `results/e4_metrics_ext.json` (`rougeL_mean`) |
| Table 11 | entity survival in the skeleton | `harness/run_ner_analysis.py` -> `results/ner_results.json` |
| Table 12 | chrF++ and METEOR | `analysis/run_e4_metrics.py` -> `results/e4_summary.json` (source `bbc`, output_kind `polished`) |
| Table 13 | rank stability across metrics | `analysis/run_e4_metrics.py` -> `results/e4_rank_stability.json`, entries `bbc_wordlevel4@<rkeep>` |
| Table 14 | human fact-level study | `python human_eval/run_e5_analysis.py --table14` (all agreement figures of Appendix B: `human_eval/run_e5_analysis.py` -> `human_eval/e5_analysis.json`) |
| Table 15 | LLM-rater grid | `human_eval/llm_raters/analyze_llm_raters.py` -> `human_eval/llm_raters/llm_rater_results.json`; calibration against the human annotators: `human_eval/llm_raters/calibrate_llm_raters.py` |
| Table 16 | Gemini 2.0 vs 2.5 Flash on the same WordFreq-3 skeletons | `analysis/run_decoder_check_wordfreq3.py` (API) -> `samples/bbc_wordfreq3_decoder_check_samples.jsonl`; `analysis/score_decoder_check.py` -> `results/decoder_check_wordfreq3_scores.json` |
| Tables 17-18 | downstream topic classification and retrieval | `analysis/run_downstream_full.py --backend tfidf --semantic` and `--backend neural --semantic` -> `results/downstream_tfidf.json`, `results/downstream_neural.json` (nine rates, both backends, with macro-F1 and Recall@5) |
| Table 19 | LLMLingua-2 as an encoder | `analysis/run_e3_llmlingua.py`, `analysis/run_e3_llmlingua_extra_rates.py` (API) -> `samples/llmlingua2_*.jsonl`; `analysis/score_e3_llmlingua_all_rates.py` -> `results/e3_llmlingua_results_all_rates.json` (Recon., Skel., realised ratio); `analysis/ner_e3_llmlingua_extra_rates.py` -> `results/e1_ner_llmlingua_all_rates.json` (Ent., Uns.) |
| Table 20 | encoder and decoder levers | `analysis/run_e2a_decoder_lever.py` -> `results/e2a_decoder_lever.json` |
| Table 21 | 3- vs 6-class WordFreq | `data/BBC_news_latest.xlsx`, sheets `wf_3classes_stats_result` and `wf_6classes_stats_result`, rows `F1_chunk_mean` and `cer_mean` |
| Table 22 | r_keep = 0.05 | `analysis/run_e0_r005.py` and `analysis/run_r005_wordfreq3.py` (API) -> `samples/bbc_hybrid_samples_r005.jsonl`, `samples/bbc_wordfreq3_samples_r005_r01.jsonl`; `analysis/score_r005_final.py` -> `results/r005_scores.json` |
| Tables 23-24 | three-tier storage accounting and regimes | `analysis/run_e6_compression.py` -> `results/e6_compression_accounting.json` (tier 1 and tier 2 with zlib; the LP row of Table 23 is strategy `opt`, see the naming map); the entity and fact bands of Table 24 come from Tables 5 and 15 |
| Table 25 | Wikipedia (WikiText) | word-level rows: `data/Wiki_text.xlsx`, sheets `benchmark_stats_result`, `adaptive_length_deletion_stats_`, `wf_3classes_stats_result`, `wf_6classes_stats_result`, `wf_optimal_stats_result`, row `F1_chunk_mean` (`notebooks/wikitext_word_level_pipeline_colab.ipynb`); semantic rows: `harness/run_appendix_semantic_hybrid.py --dataset wiki_en` (API) -> `samples/wiki_hybrid_samples.jsonl`, scored by `analysis/score_semantic_samples.py wiki`; Llama rows: `analysis/compute_wiki_llama_rows.py` (`data/SFT_qlora_Wiki_Text.xlsx`) |
| Table 26 | Reddit, BERTScore | word-level rows: `data/Reddit_Dataset.xlsx`, sheets `benchmark_stats_result`, `adaptive_length_deletion_stats_`, `wf_3classes_stats_result`, `wf_6classes_stats_result`, `wf_optimal_stats_result`, row `F1_chunk_mean` (`notebooks/reddit_word_level_pipeline_v2_colab.ipynb`); WordFreq-3 (char.) row: `analysis/compute_reddit_rows.py`; semantic rows: `harness/run_appendix_semantic_hybrid.py --dataset reddit` (API) -> `samples/reddit_hybrid_samples.jsonl`, scored by `analysis/score_semantic_samples.py reddit` (aggregates also in `results/reddit_hybrid_results.json`); Llama rows: `data/SFT_qlora_Reddit_Dataset(Testing_and_Training).xlsx`, sheets `sft_stats_result` and `base_sft_generation_stats_resul` |
| Tables 27-28 | Reddit, ROUGE-L and CER | `analysis/compute_reddit_rows.py` (its docstring states the rows and settings of each block) |
| Table 29 | Chinese news | word-level rows: `data/NEWS_Dataset.xlsx`, summary rows of sheets `STEP RESULT`, `Word Length RESULT`, `Word Frequency RESULT`, `Optimization With Polish b_i RE` (no per-row outputs are stored for these rows); semantic rows: `harness/run_appendix_semantic_hybrid.py --dataset news_zh` (API) -> `samples/news_zh_hybrid_samples.jsonl`, scored by `analysis/score_semantic_samples.py news_zh` (aggregates also in `results/news_zh_hybrid_results.json`); Qwen rows: `data/SFT_NEWS.xlsx`, sheets `sft_stats_cn_<r>` and `baseline_stats_cn_<r>` |
| Table 30 | Chinese Wikipedia | word-level rows: `data/Wiki_Dataset.xlsx`, sheets `Summary`, `Word Length Result`, `Word Frequency Result`, `Optimization With Polish b_i RE` (summary rows only); semantic rows: `--dataset wiki_zh` -> `samples/wiki_zh_hybrid_samples.jsonl`, `analysis/score_semantic_samples.py wiki_zh` (aggregates also in `results/wiki_zh_hybrid_results.json`); Qwen rows: `data/SFT_Wiki.xlsx` |
| Table 31 | Zhihu | word-level rows: `data/Zhihu_Dataset.xlsx`, sheets `Step Result`, `Word Length Result`, `Word Frequency Result`, `工作表8` (LP) (summary rows only); semantic rows: `harness/run_zhihu_hybrid_experiments.py` (API) -> `results/zhihu_hybrid_results.json` (no per-sample records were stored for this run); Qwen rows: `data/SFT_Zhihu.xlsx`, sheets `baseline_stats_cn_<r>` and `sft_stats_result_cn` |
| Table 32 | English examples | the originals are BBC `datasource` articles; the displayed compressed and reconstructed strings are single illustrative runs and are not stored in this repository |
| Table 33 | Chinese examples | `samples/news_zh_hybrid_samples.jsonl`, `sample_idx` 140 |
| Appendix P | LP objective variants | `analysis/run_lp_objective_compare.py` (API) -> `samples/bbc_lp_objective_compare_samples.jsonl`; `analysis/score_lp_objective_compare.py` -> `results/lp_objective_compare_scores.json`: the product objective of Sec. 3 (`prod`), its linear program (`sum_nop`), and the additive variant of the hybrids (`sum_p`), each used to allocate the deletion budget on 100 BBC articles and decoded with Gemini 2.5 Flash; `python analysis/run_lp_objective_compare.py --agreement` computes the LP's agreement with the product's allocation over 200 articles x 9 rates (no API) |
| Figure 1 | pipeline figure | `figures/make_pipeline_figure.py` |

`analysis/rescore_step_r06.py` recomputes the per-row scores of the Step reconstructions at r_keep = 0.6 stored in
`data/BBC_news_latest.xlsx` (sheet `benchmark`).

Notes on comparability:

- The word-level BBC tables score whole articles (n = 200-201). Tables 1, 3, 5 (semantic rows), 10, 11, 18, 19 and 22
  (semantic rows) and the semantic block of Table 15 score the 100 BBC chunks of 512 characters. The Wikipedia and Reddit semantic samples are whole
  test texts.
- English BERTScore is F1 with distilroberta-base everywhere; Chinese BERTScore uses bert-base-chinese.
- `results/e1_ner_results.json`, `results/e4_*`, `results/e6_compression_accounting.json` and
  `results/e2a_decoder_lever.json` were computed from an earlier export of the BBC workbook. It is identical to
  `data/BBC_news_latest.xlsx` in every sheet these scripts read except `benchmark` (Step), whose reconstructions were
  regenerated afterwards. Re-running them on the shipped workbook reproduces every other cell; the Step rows below
  r_keep = 0.5 (`step_char`) change, and the token-level Step rows (`step_token`) are not in the shipped workbook.

## Naming map

| Paper | Code and data |
|---|---|
| Step | `step_char` (`benchmark` sheet, columns `polished_char_<r>`), `step` in the harness |
| WordLen | `wordlen`; sheet `adpative_word_length_deletion` (sic) |
| WordFreq, WordFreq-3 / WordFreq-6 | `wordfreq3` / `wordfreq6`; sheets `wf_3classes` / `wf_6classes` |
| LP | `opt_prepolish` (sheet `wf_optimal`); in the harness `opt` |
| LP variant not reported separately | `opt` in `analysis/` (sheet `wf_optimal_without_prepolish`: bucket calibration without the second decoder pass); it is one of the strategies in the encoder spread of Table 20, and the LP row of Table 23 |
| Entropy, Entropy-LP, Entropy-in-FreqBuckets (Ent.-FreqBkt), Hybrid-alpha | `entropy`, `entropy_lp`, `entropy_in_freqbuckets`, `hybrid_alpha_<alpha>` |
| LLMLingua-2 | `llmlingua2` |
| first / second decoder pass of the word-level pipeline | `restored_*` / `polished_*` columns and output kinds; the paper scores the second pass |
| QLoRA (SFT) / zero-shot Llama on BBC | sources `bbc_sft_qlora` / `bbc_sft_base`, strategy key `wordfreq_opt_sft` (per-rate best strategy: LP at 0.9 and 0.5, WordLen at 0.7, WordFreq-3 at 0.3 and 0.1) |
| unsupported-entity rate | `unsupported_entity_rate` (`unsupported_rate`, `unsupported` in some score files) |
| invented claim (human and LLM raters) | `hallucinated_claim_present` / `halluc` in the annotation files |
| LP objectives (Appendix P) | `prod` (Sec. 3), `sum_nop` (its linear program), `sum_p` (additive variant of the hybrids) |

Script prefixes: E0 = BBC semantic and hybrid grid with per-sample records; E1 = entity metrics on the
reconstructions; E2a = encoder and decoder levers; E3 = LLMLingua-2; E4 = lexical metrics and rank stability;
E5 = human fact-level study; E6 = storage accounting.

## Data

| Workbook | Content |
|---|---|
| `data/BBC_news_latest.xlsx` | BBC News: `datasource` (2,228 articles with category and summary) and, per word-level strategy, skeletons, reconstructions (both decoder passes), per-row scores and stats sheets; Llama SFT and zero-shot outputs (`sft_*`, `base_sft_*`); sheet descriptions in `whole_datasheet_description` |
| `data/Wiki_text.xlsx` | WikiText (English Wikipedia), word-level strategies |
| `data/SFT_qlora_Wiki_Text.xlsx` | WikiText fine-tuning splits and Llama base / SFT generations |
| `data/Reddit_Dataset.xlsx` | Reddit comments, word-level strategies |
| `data/SFT_qlora_Reddit_Dataset(Testing_and_Training).xlsx` | Reddit fine-tuning splits and Llama base / SFT generations |
| `data/NEWS_Dataset.xlsx` | Chinese official news: test and training texts, HanLP segmentation, word-level results |
| `data/Wiki_Dataset.xlsx` | Chinese Wikipedia, as above |
| `data/Zhihu_Dataset.xlsx` | Zhihu answers, as above |
| `data/SFT_NEWS.xlsx`, `data/SFT_Wiki.xlsx`, `data/SFT_Zhihu.xlsx` | Qwen2.5-3B-Instruct base / SFT results on the three Chinese datasets |

The workbooks contain the source texts of public corpora in the form our experiments read them: BBC News
(Greene and Cunningham, 2006, via the BBC News Summary dataset), WikiText (Merity et al., 2017), a Reddit comment
dataset, Chinese official news, Chinese Wikipedia, and Zhihu answers, all cited in the paper. They are provided for
research reproduction only and remain under the terms of the original datasets. The MIT license in `LICENSE`
covers the code.

## Running

- Python 3.10+: `pip install -r requirements.txt`, then `python -m spacy download en_core_web_sm` and
  `python -c "import nltk; nltk.download('wordnet')"` (METEOR).
- Gemini scripts read `GOOGLE_API_KEY` from the environment or from `~/.env` (`GOOGLE_API_KEY=...`); the LLM raters
  need `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`. No key is stored in this repository.
- The decoder of the published word-level and cross-domain runs, gemini-2.0-flash, has been retired
  by its provider. The harness scripts now default to gemini-2.5-flash; the scripts of the later experiments
  (Tables 1, 3, 16, 19, 22 and the BBC semantic grid) used gemini-2.5-flash. Table 16 compares the two decoders on the
  same skeletons.
- The notebooks are Google Colab notebooks that read and write Google Drive and Google Sheets; see their first cell.
- BERTScore runs on CPU, CUDA or Apple MPS.

## Adapter weights

The eleven QLoRA adapters for Llama-3.2-3B-Instruct (`adapter_model.safetensors`, 194.6 MB each) are not in this
repository; they are available from the authors on request (contact details in the paper). `sft/adapters/` holds
every config and split needed to retrain them.

## Citation

See `CITATION.cff`.
