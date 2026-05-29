from setuptools import setup, find_packages


required_packages = [
    'pyyaml',
    'pydantic',
    'pytest',
    'numpy',
    'pandas',
    'scipy'
]


setup(
    name='copper_usage', 
    author='Johannes Grygier, Beca Liang',
    version='0.1.0',
    package_dir={'': 'src'},
    packages=find_packages(
        where='src',
        include=[
            'copper_usage',
            'copper_usage.*',
        ],
    ),
    install_requires=required_packages
)
