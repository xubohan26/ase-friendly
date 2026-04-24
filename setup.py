from setuptools import setup, find_packages

setup(
    name='ase-friendly',
    version='0.1.0',
    description='A zero-learning-curve prompt-based CLI tool for crystal structure manipulation',
    author='Bohan Xu',
    packages=find_packages(), 
    install_requires=[
        'ase',
        'networkx',
        'numpy',
        'pandas',
        'scipy'
    ],
    entry_points={
        'console_scripts': [
            'ase-friendly=ase_friendly.main:main', 
        ],
    },
)