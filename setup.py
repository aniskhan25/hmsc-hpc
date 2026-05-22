from setuptools import setup, find_packages

setup(
    name="hmsc",
    version="0.1.8",
    author="[removed for review]",
    license='GPLv3+',
    packages=find_packages(include=['hmsc', 'hmsc.*']),
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
        'rds': ['pyreadr'],
        'diagnostics': ['arviz'],
    },
)
