# Python-Native Tutorial

Compile and sample without R:

```bash
python -m pyhmsc compile examples/projects/fixed_poisson/model.yaml --output run
python -m pyhmsc sample run/init.json --output run/posterior.h5 --samples 100 --transient 100 --thin 1
python -m pyhmsc summarize run/posterior.h5 --param Beta
```

From Python:

```python
import pandas as pd
from pyhmsc import HmscModel

Y = pd.read_csv("examples/projects/fixed_poisson/data/Y.csv", index_col=0)
X = pd.read_csv("examples/projects/fixed_poisson/data/X.csv", index_col=0)

model = HmscModel(Y=Y, X=X, x_formula="~ forest_cover + elevation", distr="poisson")
fit = model.sample(samples=100, transient=100, thin=1, chains=2, init="python-native")
print(fit.beta_mean())
```
