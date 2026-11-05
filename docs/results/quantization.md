| Model | Size (MB) | Accuracy | Macro F1 | Median ms (batch 1) | p95 ms | Speed-up |
|---|---|---|---|---|---|---|
| fp32 | 44.72 | 0.9795 | 0.9788 | 8.12 | 40.32 | 1.00x |
| int8 dynamic | 11.24 | 0.9798 | 0.9791 | 181.19 | 231.38 | 0.04x |
| int8 static | 11.3 | 0.9780 | 0.9773 | 2.43 | 5.53 | 3.34x |
