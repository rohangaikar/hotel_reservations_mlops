from setuptools import setup,find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name = 'HOTEL-RESERVATION-PREDICTION',
    version = "0.1",
    author = 'rvg93',
    packages = find_packages(),
    install_requires = requirements,
)