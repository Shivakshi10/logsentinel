| Model | Precision | Recall | F1 | ROC-AUC | Incidents caught | Avg. delay (min) | False alarms/day |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static error-rate threshold | 0.85 | 0.44 | 0.58 | 0.774 | 8/12 | 2.2 | 9.0 |
| Isolation Forest | 0.40 | 0.64 | 0.49 | 0.878 | 11/12 | 1.5 | 88.0 |
| Isolation Forest + novelty rule | 0.46 | 0.85 | 0.60 | 0.878 | 12/12 | 0.9 | 88.0 |
| LSTM Autoencoder | 0.96 | 0.91 | 0.93 | 0.965 | 12/12 | 0.8 | 4.0 |
| LSTM Autoencoder + novelty rule | 0.96 | 0.91 | 0.93 | 0.965 | 12/12 | 0.8 | 4.0 |
