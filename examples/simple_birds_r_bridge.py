import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyhmsc import HmscModel


Y = pd.DataFrame(
    {
        "sparrow": [3, 5, 0, 1, 4],
        "owl": [0, 0, 2, 1, 0],
        "woodpecker": [1, 2, 1, 0, 3],
    }
)
X = pd.DataFrame(
    {
        "forest_cover": [80, 70, 20, 40, 90],
        "elevation": [200, 250, 800, 600, 150],
    }
)

model = HmscModel(
    Y=Y,
    X=X,
    x_formula="~ forest_cover + elevation",
    distr="poisson",
)
fit = model.sample(samples=100, transient=200, thin=2, chains=4, init="r-bridge")
print(fit.summary("Beta"))

new_X = pd.DataFrame({"forest_cover": [30, 85], "elevation": [700, 180]})
print(fit.predict(new_X))
