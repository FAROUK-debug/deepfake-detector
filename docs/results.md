# Results

Validation images used to pick the threshold: 1000 (balanced). Test images: 2000 (balanced, different from validation).

**AUC: 0.941**, inference on CPU: 108 ms per image (batches of 32).

| Metric | Threshold 0.5 | Threshold 0.27 (from validation) |
|---|---|---|
| accuracy | 86.0% | 86.8% |
| precision (fake) | 82.0% | 88.3% |
| recall (fake) | 92.2% | 84.7% |
| f1 (fake) | 86.8% | 86.5% |

Confusion matrix at threshold 0.27 (rows = true class):

| | predicted REAL | predicted FAKE |
|---|---|---|
| **true REAL** | 888 | 112 |
| **true FAKE** | 153 | 847 |
