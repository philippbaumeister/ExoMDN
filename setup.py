import os
from setuptools import setup

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ExoMDN",
    version="1.0.0",
    packages=["exomdn"],
    url="https://github.com/philippbaumeister/ExoMDN",
    license="MIT",
    author="Philipp Baumeister",
    author_email="philipp.baumeister@dlr.de",
    description="Rapid characterization of exoplanet interiors with Mixture Density Networks",
    long_description=long_description
    )
