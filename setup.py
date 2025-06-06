from setuptools import setup, find_packages


INSTALL_REQUIRES = [
    'cycler>=0.12.1',
    'pandas>=2.1.4',
    'pyarrow>=15.0.0',  # To suppress pandas future warning
    'pingouin>=0.5.3',
    'matplotlib>=3.8.2',
    'seaborn>=0.13.1',
    'tqdm>=4.65.0',
    'dynaconf>=3.2.10',
]

setup(
    name='issych',
    version='0.0.6.1',
    author='Issaku Kawashima',
    author_email='issakuss@gmail.com',
    description='Convenient tools for psychological studies',
    url='https://github.com/issakuss/issych',
    packages=find_packages(),
    license='MIT',
    install_requires=INSTALL_REQUIRES
)
