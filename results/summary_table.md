# NoReC binary sentiment - test-set results (mean ± std across seeds)

| family | model | config | n_seeds | accuracy | macro_f1 | neg_f1 | pos_f1 | n_unparseable |
|---|---|---|---|---|---|---|---|---|
| baseline | majority | always positive | 1 | 0.7753 | 0.4367 | 0.0000 | 0.8735 |  |
| classical | Logistic Regression | class_weight=none, C=10.0 | 1 | 0.8717 | 0.7907 | 0.6606 | 0.9209 |  |
| classical | Logistic Regression | class_weight=balanced, C=10.0 | 1 | 0.8765 | 0.8190 | 0.7170 | 0.9210 |  |
| classical | Linear SVM | class_weight=none, C=10.0 | 1 | 0.8712 | 0.7995 | 0.6797 | 0.9194 |  |
| classical | Linear SVM | class_weight=balanced, C=1.0 | 1 | 0.8747 | 0.8142 | 0.7082 | 0.9202 |  |
| transformer | NB-BERT-base | 3 ep, lr=2e-05, batch=32, max_len=512 | 5 | 0.8994 ± 0.0033 | 0.8529 ± 0.0038 | 0.7703 ± 0.0054 | 0.9356 ± 0.0023 |  |
| transformer | NB-BERT-base + chunk-and-pool | max_len=512, stride=256 | 1 | 0.9030 | 0.8544 | 0.7703 | 0.9385 |  |
| LLM (few-shot) | Llama-3.1-8B-Instruct | few-shot | 5 | 0.9131 ± 0.0044 | 0.8777 ± 0.0036 | 0.8119 ± 0.0045 | 0.9435 ± 0.0035 | 42 |
| LLM (few-shot) | Llama-3.1-8B-Instruct | few-shot, prompt=norwegian | 1 | 0.9111 | 0.8788 | 0.8162 | 0.9413 | 17 |
| LLM (few-shot) | Llama-3.1-8B-Instruct | few-shot, prompt=terse | 1 | 0.9141 | 0.8775 | 0.8106 | 0.9444 | 109 |
| LLM (zero-shot) | Llama-3.1-8B-Instruct | zero-shot | 1 | 0.8733 | 0.8343 | 0.7540 | 0.9146 | 92 |
| LLM (few-shot) | Llama-3.2-1B-Instruct | few-shot | 5 | 0.7397 ± 0.0594 | 0.6959 ± 0.0429 | 0.5838 ± 0.0310 | 0.8080 ± 0.0569 | 0 |
| LLM (zero-shot) | Llama-3.2-1B-Instruct | zero-shot | 1 | 0.2859 | 0.2667 | 0.3855 | 0.1479 | 9 |
| LLM (few-shot) | Llama-3.2-3B-Instruct | few-shot | 5 | 0.8705 ± 0.0188 | 0.8311 ± 0.0174 | 0.7496 ± 0.0200 | 0.9125 ± 0.0148 | 0 |
| LLM (zero-shot) | Llama-3.2-3B-Instruct | zero-shot | 1 | 0.8899 | 0.8436 | 0.7586 | 0.9287 | 0 |
