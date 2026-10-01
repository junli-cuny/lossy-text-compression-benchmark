# QLoRA decoder adapters (Llama-3.2-3B-Instruct)

`adapters/` holds, for every adapter behind a Llama SFT number in the paper, its `adapter_config.json` and
`tokenizer_config.json` and the train / validation / test splits it was trained and evaluated on
(`repair_sft_*_{train,val,test}_<rate>.json`, chat-messages format: the user turn carries the instruction and
the skeleton, the assistant turn the original text). The adapter weights (`adapter_model.safetensors`,
194.6 MB each) are not in the repository; see the top-level README.

| Adapter directory | Strategy, r_keep | Paper |
|---|---|---|
| `SFT_BBC_News/best_sft_wf_optimal_50_90/qlora_llama3_decompress_wf_opt_90_lr2e4_lora` | LP, 0.9 | Table 6 |
| `SFT_BBC_News/best_sft_wld_70/qlora_llama3_decompress_wld_70_lr2e4_lora` | WordLen, 0.7 | Table 6 |
| `SFT_BBC_News/best_sft_wf_optimal_50_90/qlora_llama3_decompress_wf_opt_50_lr2e4_lora` | LP, 0.5 | Table 6 |
| `SFT_BBC_News/best_sft_wf_3classes_10_30/qlora_llama3_decompress_wf_3c_30_lr2e4_lora` | WordFreq-3, 0.3 | Table 6 |
| `SFT_BBC_News/best_sft_wf_3classes_10_30/qlora_llama3_decompress_wf_3c_10_lr2e4_lora` | WordFreq-3, 0.1 | Table 6 |
| `SFT_Wiki/best_sft_wld_90_70_50/qlora_llama3_decompress_wld_{90,70,50}_lr2e4_lora` | WordLen, 0.9 / 0.7 / 0.5 | Table 25 |
| `SFT_Wiki/best_sft_wf_3classes_30_10/qlora_llama3_decompress_wf_3c_{30,10}_lr2e4_lora` | WordFreq-3, 0.3 / 0.1 | Table 25 |
| `SFT_Reddit_Comment/best_sft_50/qlora_llama3_decompress_50_lr2e4_lora` | r_keep 0.5 | Table 26 (r_keep = 0.5 cell) |

The adapters of the other Reddit rates are not included.

## Where the paper's numbers come from

The fine-tuning and its evaluation were run in the notebooks (`notebooks/bbc_word_level_pipeline_v2.ipynb`,
`notebooks/wikitext_word_level_pipeline_colab.ipynb`, `notebooks/reddit_word_level_pipeline_v2_colab.ipynb`), and
the published cells are stored in the workbooks: Table 6 in `data/BBC_news_latest.xlsx`, sheets
`sft_generation_stats_result` (SFT, row `F1_chunk`) and `base_sft_stats_result` (zero-shot Llama, row `F1_chunk`);
the Wikipedia Llama rows of Table 25 in `data/SFT_qlora_Wiki_Text.xlsx` (`analysis/compute_wiki_llama_rows.py`
recomputes them); the Reddit Llama rows of Table 26 in `data/SFT_qlora_Reddit_Dataset(Testing_and_Training).xlsx`.

## Re-evaluation script

`eval_sft_adapters.py` is a separate re-evaluation, not the source of the published cells. For every adapter
directory under `--adapter-root` it reconstructs the adapter's held-out test split (found in the parent directory
by its rate) with Llama-3.2-3B-Instruct plus the adapter (4-bit NF4 base by default, greedy decoding) and, with
`--include-base`, with the base model alone on the same inputs, and scores both with BERTScore F1
(distilroberta-base) and ROUGE-L. Copy the adapter weights into their directories first, then for example:

    python sft/eval_sft_adapters.py --adapter-root sft/adapters/SFT_BBC_News --out-dir sft/results/SFT_BBC_News --include-base

`results/<dataset>/` holds the per-sample reconstructions of one such re-evaluation (`<adapter>.jsonl` and
`<adapter>__base.jsonl`), run with an unquantized fp16 base (`--quant bf16 --dtype float16`). Its setup differs from
the notebooks' (greedy decoding, no second decoder pass), so its scores are not the published cells; in particular the base model often
declines the reconstruction prompt and scores lower.
