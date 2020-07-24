import ah_bootstrap
from setuptools import setup, find_packages

setup_requires = []
install_requires = ['numpy',
                    'astropy >= 2.0',
                    'bottleneck']

classifiers = ["Intended Audience :: Science/Research"]

setup(
    name="imcombinepy",
    version="0.0.1",
    author="Yoonsoo P. Bach",
    author_email="dbstn95@gmail.com",
    description="",
    license="",
    keywords="",
    url="",
    classifiers=classifiers,
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=install_requires)
