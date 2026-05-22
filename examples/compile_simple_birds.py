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
    },
    index=["plot1", "plot2", "plot3", "plot4", "plot5"],
)
X = pd.DataFrame(
    {
        "forest_cover": [80, 70, 20, 40, 90],
        "elevation": [200, 250, 800, 600, 150],
    },
    index=Y.index,
)

model = HmscModel(
    Y=Y,
    X=X,
    x_formula="~ forest_cover + elevation",
    distr="poisson",
)
compiled = model.compile("run_001", chains=4)
print(compiled.init_json)
