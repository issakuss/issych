from setuptools import setup, find_packages


INSTALL_REQUIRES = [
    'numpy>=1.23.5',
    'pandas>=1.5.3'
]

setup(
    name='issych',
    version='0.0.1',
    author='Issaku Kawashima',
    author_email='issakuss@gmail.com',
    description='Convenient tools for psychological studies',
    url='https://github.com/issakuss/issych',
    packages=find_packages(),
    license='MIT',
    install_requires=INSTALL_REQUIRES
)