from setuptools import setup, find_packages

setup(
    name="hmsc-hpc",
    version="0.1.8",
    author="[removed for review]",
    license='GPLv3+',
    packages=find_packages(include=['hmsc', 'hmsc.*', 'pyhmsc', 'pyhmsc.*']),
    install_requires=[
        'numpy',
        'pandas',
        'patsy',
        'h5py',
        'PyYAML',
        'scipy',
        'tensorflow',
        'tensorflow-probability[tf]',
        'ujson',
    ],
    extras_require={
        'parquet': ['pyarrow'],
        'zarr': ['zarr'],
        'phylo': ['biopython'],
        'rds': ['pyreadr'],
        'diagnostics': ['arviz'],
    },
)
