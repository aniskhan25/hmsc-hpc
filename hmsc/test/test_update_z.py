import pytest
import numpy as np
import tensorflow as tf

from numpy import nan
from pytest import approx

from hmsc.updaters.updateZ import updateZ
from hmsc.test import convert_to_tf


SEED = 42


def run_test(input_values, ref_values, *,
             tnlib='tf',
             poisson_preupdate_z=False,
             poisson_marginalize_z=False,
             ):
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    # Convert inputs to tf
    params, data, rLHyperparams = map(convert_to_tf, input_values)
    dtype = data['Y'].dtype
    assert dtype == tf.float64

    # Calculate
    Z, iD, omega = updateZ(
        params, data, rLHyperparams,
        poisson_preupdate_z=poisson_preupdate_z,
        poisson_marginalize_z=poisson_marginalize_z,
        truncated_normal_library=tnlib,
        dtype=dtype,
        )
    Z = Z.numpy()
    iD = iD.numpy()
    omega = omega.numpy()

    assert Z.shape == params['Z'].shape
    assert iD.shape == params['Z'].shape
    assert omega.shape == params['poisson_omega'].shape

    # Print values
    print()
    for name, array in (('Z', Z),
                        ('iD', iD),
                        ('omega', omega)):
        print(f'    {name} = \\')
        print(np.array2string(array, separator=', ', max_line_width=200))

    # Test against reference
    ref_Z, ref_iD, ref_omega = map(np.asarray, ref_values)
    assert Z == approx(ref_Z, nan_ok=True)
    assert iD == approx(ref_iD)
    if omega.size > 0 or ref_omega.size > 0:
        assert omega == approx(ref_omega, nan_ok=True)


def default_input_values(rng):
    ns = 7
    ny = 5
    nb = 11
    nr = 2
    ne = 3

    distr = np.array(
        [[1, 1],
         [2, 1],
         [3, 1],
         [1, 1],
         [3, 1],
         [2, 1],
         [1, 1]],
        dtype=np.int32,
    )

    Pi = np.array(
        [[0, 2],
         [1, 2],
         [2, 2],
         [0, 0],
         [1, 0]],
        dtype=np.int32,
    )

    ns3 = np.sum(distr[:, 0] == 3)

    params = dict(
        Z=rng.random(size=(ny, ns)),
        Beta=rng.random(size=(nb, ns)),
        sigma=rng.random(size=ns),
        Xeff=rng.random(size=(ny, nb)),
        poisson_omega=rng.random(size=(ny, ns3)),
    )
    data = dict(
        Y=rng.integers(2, size=(ny, ns)).astype(np.float64),
        Pi=Pi,
        distr=distr,
    )

    assert data['distr'].shape == (ns, 2)
    assert data['Pi'].shape == (ny, nr)

    EtaList = [
        rng.random(ne * 2).reshape(ne, 2),
        rng.random(ne * 2).reshape(ne, 2),
    ]
    LambdaList = [
        rng.random(2 * ns).reshape(2, ns),
        rng.random(2 * ns).reshape(2, ns),
    ]
    rLHyperparams = [
        dict(xDim=0),
        dict(xDim=0),
    ]
    assert len(EtaList) == len(LambdaList) == len(rLHyperparams) == nr

    params["Eta"] = EtaList
    params["Lambda"] = LambdaList

    return params, data, rLHyperparams


def default_reference_values():
    Z = \
[[ 0.        ,  2.53617218,  1.40754338,  1.        ,  0.13129104, -0.02721602,  1.        ],
 [ 0.        ,  3.0099937 ,  1.35676311,  0.        ,  0.89406079, -0.00651452,  1.        ],
 [ 1.        , -0.03079559,  1.77882987,  0.        ,  0.6501287 ,  3.00193797,  1.        ],
 [ 0.        ,  2.7197177 ,  1.68541477,  0.        ,  0.11587453,  3.92979457,  1.        ],
 [ 0.        , -0.05136673,  1.44970694,  0.        ,  0.32676159,  4.30543888,  0.        ]]
    iD = \
[[10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  5.35606031]]
    omega = \
[[82.69564925, 73.52446685],
 [75.73119083, 80.31543575],
 [77.77786621, 79.68186375],
 [81.55596303, 73.14161346],
 [80.27240777, 75.35136998]]
    return Z, iD, omega


def test_defaults():
    rng = np.random.default_rng(seed=SEED)
    run_test(
        default_input_values(rng),
        default_reference_values(),
    )


@pytest.mark.parametrize("tnlib", ['tf', 'tfd'])
def test_tnlib(tnlib):
    rng = np.random.default_rng(seed=SEED)
    params, data, rLHyperparams = default_input_values(rng)
    Z, iD, omega = default_reference_values()

    if tnlib == 'tfd':
        Z = \
[[ 0.        ,  2.17921306,  1.40754338,  1.        ,  0.13129104, -0.05694204,  1.        ],
 [ 0.        ,  2.7920181 ,  1.35676311,  0.        ,  0.89406079, -0.13361183,  1.        ],
 [ 1.        , -0.0166318 ,  1.77882987,  0.        ,  0.6501287 ,  4.70081388,  1.        ],
 [ 0.        ,  4.68685993,  1.68541477,  0.        ,  0.11587453,  4.4379023 ,  1.        ],
 [ 0.        , -0.1906678 ,  1.44970694,  0.        ,  0.32676159,  2.69563436,  0.        ]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
        tnlib=tnlib,
    )


def test_y_nan():
    rng = np.random.default_rng(seed=SEED)
    params, data, rLHyperparams = default_input_values(rng)
    Z, iD, omega = default_reference_values()

    Y = rng.integers(3, size=data['Y'].shape).astype(np.float64)
    Y[Y == 2] = nan
    data['Y'] = Y

    Z = \
[[ 2.6390101 ,  2.53617218,  1.40754338,  1.        ,  0.14450654,  2.75723293,  1.        ],
 [ 4.25794634,  3.0099937 ,          0.,  1.        ,  0.89406079,  3.13420971,  0.        ],
 [ 1.        ,  3.42304933,  1.78706598,  0.        ,  0.66243985,  3.00193797,  4.65326605],
 [ 0.        ,  2.7197177 ,  1.6935875 ,  0.        ,  0.1025117 , -0.05852222,  3.19747972],
 [ 3.36759254,  3.50740932,          0.,  3.19999282,  0.31382702,  4.30543888,  0.        ]]
    iD = \
[[ 0.        ,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  0.        ,  5.35606031],
 [ 0.        ,  0.        ,  0.        ,  1.36279137,  1.73806681,  1.93189325,  5.35606031],
 [10.68268147,  0.        , 32.00141647,  1.36279137,  1.73806681,  0.        ,  0.        ],
 [10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  1.73806681,  0.        ,  5.35606031]]
    omega = \
[[82.69564925, 73.59783994],
 [        nan, 80.31543575],
 [77.85596886, 79.76263482],
 [81.63810387, 73.06792919],
 [        nan, 75.27585429]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
    )


@pytest.mark.parametrize("distr_case", [1, 2, 3])
def test_distr_case(distr_case):
    rng = np.random.default_rng(seed=SEED)
    params, data, rLHyperparams = default_input_values(rng)
    Z, iD, omega = default_reference_values()

    if distr_case in [1, 2, 3]:
        data['distr'][:, 0] = distr_case
        ns3 = np.sum(data['distr'][:, 0] == 3)
        ny = params['poisson_omega'].shape[0]
        params['poisson_omega'] = rng.random(size=(ny, ns3))

    if distr_case == 1:
        Z = \
[[0., 1., 0., 1., 0., 0., 1.],
 [0., 1., 1., 0., 0., 0., 1.],
 [1., 0., 0., 0., 0., 1., 1.],
 [0., 1., 0., 0., 1., 1., 1.],
 [0., 0., 0., 0., 1., 1., 0.]]
    elif distr_case == 2:
        Z = \
[[-3.49037857e-02,  1.87095934e+00, -2.28259734e-03,  1.08649303e+00, -4.60635976e-01, -2.99549625e-02,  2.43555209e+00],
 [-4.77320861e-03,  2.59930013e+00,  3.76303487e+00, -2.37961062e-02, -1.60768148e-01, -1.48271971e-02,  2.97103997e+00],
 [ 3.65624205e+00, -1.28451487e-01, -2.97511988e-02, -9.53287325e-02, -7.88738335e-02,  3.82215867e+00,  4.21878878e+00],
 [-1.90880032e-02,  2.22307589e+00, -7.47232999e-04, -1.83280665e-01,  2.64592351e+00,  3.47088769e+00,  2.51095077e+00],
 [-4.55057056e-02, -2.37860572e-01, -1.06874815e-03, -6.50872312e-02,  4.07241451e+00,  3.31299907e+00, -2.41794754e-02]]
    elif distr_case == 3:
        Z = \
[[0.92624665, 0.45003419, 1.31984705, 0.80502735, 0.19884378, 0.9822903 , 0.72215214],
 [0.92802478, 0.32279442, 1.465807  , 0.19966078, 0.80088564, 0.41636724, 1.12555882],
 [0.85577721, 0.35917522, 1.72506687, 0.33495094, 1.04227898, 0.55410351, 1.09129144],
 [0.57492908, 0.97391634, 1.71698628, 0.69326334, 0.22491319, 0.52106726, 0.12154811],
 [0.51790385, 0.92475001, 1.34057843, 0.97366586, 0.52101259, 0.67950832, 0.99466291]]

    if distr_case in [1, 2]:
        omega = \
[]
    elif distr_case == 3:
        omega = \
[[79.19229817, 78.03413752, 81.12608371, 82.29633972, 73.30738476, 82.86186661, 79.14927862],
 [78.53548146, 74.97353313, 79.98110283, 75.6570093 , 83.75325274, 77.44605529, 82.75470035],
 [76.7403655 , 73.97389438, 77.9654872 , 74.31207503, 82.07980997, 78.44424682, 80.38287021],
 [76.03236043, 81.88722941, 82.05959659, 80.71769938, 74.73541972, 77.01701888, 72.11164144],
 [73.96866169, 80.25482844, 78.88315942, 85.25988827, 77.12761621, 80.0202054 , 77.72359649]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
    )


def test_x_ndim_3():
    rng = np.random.default_rng(seed=SEED)
    params, data, rLHyperparams = default_input_values(rng)
    Z, iD, omega = default_reference_values()

    _, ns = params['Z'].shape
    ny, nb = params['Xeff'].shape
    params['Xeff'] = rng.random(size=(ns, ny, nb))

    Z = \
[[ 0.        ,  3.53200434,  1.75342901,  1.        ,  0.13864651, -0.01864537,  1.        ],
 [ 0.        ,  2.68074041,  1.43855754,  0.        ,  0.87347456, -0.00771769,  1.        ],
 [ 1.        , -0.02915212,  1.72133414,  0.        ,  0.64247312,  2.95720883,  1.        ],
 [ 0.        ,  2.40970493,  1.57816434,  0.        ,  0.14337646,  3.19505106,  1.        ],
 [ 0.        , -0.05266042,  1.63378067,  0.        ,  0.29615809,  4.05959123,  0.        ]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
    )


@pytest.mark.parametrize("preupdate", [True, False])
@pytest.mark.parametrize("marginalize", [True, False])
def test_poisson_flags(preupdate, marginalize):
    rng = np.random.default_rng(seed=SEED)
    params, data, rLHyperparams = default_input_values(rng)
    Z, iD, omega = default_reference_values()

    if preupdate and marginalize:
        Z = \
[[ 0.00000000e+00,  2.53617218e+00, -1.26355164e+01,  1.00000000e+00, -2.77105558e+02, -2.72160232e-02,  1.00000000e+00],
 [ 0.00000000e+00,  3.00999370e+00, -1.17278655e+01,  0.00000000e+00, -1.79726207e+02, -6.51451591e-03,  1.00000000e+00],
 [ 1.00000000e+00, -3.07955853e-02, -1.04428030e+01,  0.00000000e+00, -1.97247049e+02,  3.00193797e+00,  1.00000000e+00],
 [ 0.00000000e+00,  2.71971770e+00, -1.12510798e+01,  0.00000000e+00, -1.85920900e+02,  3.92979457e+00,  1.00000000e+00],
 [ 0.00000000e+00, -5.13667342e-02, -1.19154864e+01,  0.00000000e+00, -2.39028750e+02,  4.30543888e+00,  0.00000000e+00]]
    elif not preupdate and marginalize:
        Z = \
[[ 0.        ,  2.53617218,  0.86148798,  1.        ,  0.10729794, -0.02721602,  1.        ],
 [ 0.        ,  3.0099937 ,  0.31205812,  0.        ,  0.68230191, -0.00651452,  1.        ],
 [ 1.        , -0.03079559,  0.47919116,  0.        ,  0.63280165,  3.00193797,  1.        ],
 [ 0.        ,  2.7197177 ,  0.77699572,  0.        ,  0.0785376 ,  3.92979457,  1.        ],
 [ 0.        , -0.05136673,  0.67896492,  0.        ,  0.27881144,  4.30543888,  0.        ]]
    elif preupdate and not marginalize:
        Z = \
[[ 0.00000000e+00,  2.53617218e+00, -4.06070277e+00,  1.00000000e+00, -1.37245984e+02, -2.72160232e-02,  1.00000000e+00],
 [ 0.00000000e+00,  3.00999370e+00, -3.14339495e+00,  0.00000000e+00, -1.06936268e+02, -6.51451591e-03,  1.00000000e+00],
 [ 1.00000000e+00, -3.07955853e-02, -2.35608505e+00,  0.00000000e+00, -1.13780474e+02,  3.00193797e+00,  1.00000000e+00],
 [ 0.00000000e+00,  2.71971770e+00, -2.94979503e+00,  0.00000000e+00, -1.10791375e+02,  3.92979457e+00,  1.00000000e+00],
 [ 0.00000000e+00, -5.13667342e-02, -3.30015786e+00,  0.00000000e+00, -1.27190998e+02,  4.30543888e+00,  0.00000000e+00]]

    if preupdate and marginalize:
        iD = \
[[10.68268147,  2.98066761, 14.21764018,  1.36279137,  0.87460102,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 14.58636705,  1.36279137,  1.05416311,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 15.16306357,  1.36279137,  1.0166099 ,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 14.80028053,  1.36279137,  1.04015494,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 14.51482035,  1.36279137,  0.93657754,  1.93189325,  5.35606031]]
    elif not preupdate and marginalize:
        iD = \
[[10.68268147,  2.98066761, 23.07276037,  1.36279137,  1.69792896,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 22.49556043,  1.36279137,  1.70125087,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 22.67278332,  1.36279137,  1.70096439,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 22.98315046,  1.36279137,  1.69772374,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 22.88005036,  1.36279137,  1.69888016,  1.93189325,  5.35606031]]

    if preupdate:
        omega = \
[[25.58425259,  1.76048085],
 [26.80350743,  2.67904081],
 [28.81751649,  2.44912189],
 [27.53480601,  2.59038264],
 [26.5629061 ,  2.03101203]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
        poisson_preupdate_z=preupdate,
        poisson_marginalize_z=marginalize,
    )
