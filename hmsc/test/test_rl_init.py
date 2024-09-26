import numpy as np
import tensorflow as tf

from pytest import approx

from hmsc.utils.import_utils import calculate_GPP
from hmsc.test import convert_to_tf


SEED = 42


def input_values(rng):
    na = 2
    n1 = 4
    n2 = 3

    alpha = rng.random(na)
    alpha[-1] = 0.0
    d12 = rng.random(n1 * n2).reshape(n1, n2)
    d22 = rng.random(n2 * n2).reshape(n2, n2)
    d22 = 0.5 * (d22 + d22.T)
    np.fill_diagonal(d22, 0)
    return d12, d22, alpha


def reference_values():
    idD = \
[[ 4.74732221,  1.21841991, 16.81039263,  1.27917015],
 [ 1.        ,  1.        ,  1.        ,  1.        ]]
    iDW12 = \
[[[ 1.56552007,  1.92810424,  4.20341649],
  [ 0.3454143 ,  0.45571594,  0.44127379],
  [14.2458611 ,  9.3939915 , 10.41141587],
  [ 0.38626669,  0.55671448,  0.44182194]],

 [[ 0.        ,  0.        ,  0.        ],
  [ 0.        ,  0.        ,  0.        ],
  [ 0.        ,  0.        ,  0.        ],
  [ 0.        ,  0.        ,  0.        ]]]
    F = \
[[[13.80338756,  9.72259845, 10.89601792],
  [ 9.72259845,  7.44538439,  8.4114639 ],
  [10.89601792,  8.4114639 , 11.48249436]],

 [[ 1.        ,  0.        ,  0.        ],
  [ 0.        ,  1.        ,  0.        ],
  [ 0.        ,  0.        ,  1.        ]]]
    iF = \
[[[ 0.90649386, -1.22934882,  0.04036143],
  [-1.22934882,  2.44625374, -0.62543626],
  [ 0.04036143, -0.62543626,  0.50695045]],

 [[ 1.        ,  0.        ,  0.        ],
  [ 0.        ,  1.        ,  0.        ],
  [ 0.        ,  0.        ,  1.        ]]]
    detD = \
[-0.54608872,  0.        ]
    return idD, iDW12, F, iF, detD


def test_calculate_GPP():
    rng = np.random.default_rng(seed=SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    inputs = input_values(rng)
    inputs = map(convert_to_tf, inputs)

    values = calculate_GPP(*inputs)
    values = list(map(lambda a: a.numpy(), values))
    names = ['idD', 'iDW12', 'F', 'iF', 'detD']
    assert len(names) == len(values)

    # Print values
    print()
    for name, array in zip(names, values):
        print(f'    {name} = \\')
        print(np.array2string(array, separator=', ', max_line_width=200))

    # Test against reference
    ref_values = list(map(np.asarray, reference_values()))
    assert len(ref_values) == len(values)
    for name, val, ref in zip(names, values, ref_values):
        assert val == approx(ref), name
