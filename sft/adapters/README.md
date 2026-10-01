Adapter configs and the train / validation / test splits of each QLoRA run (chat-messages format). Each adapter
directory holds `adapter_config.json` and `tokenizer_config.json`; its splits sit in the parent directory and carry
the retention rate in their name. The adapter weights (`adapter_model.safetensors`, 194.6 MB each) are not in the
repository; place them in the adapter directory to run `../eval_sft_adapters.py`. See `../README.md`.
