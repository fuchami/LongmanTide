# -*- coding: utf-8 -*-

from setuptools import setup

__version__ = '0.1.1'
__description__ = """
Longman Tide is a small python library used to compute the accelerations at a given location and                   
time on earth, due to the tidal effects of the Sun and Moon. The calculated accelerations can then be 
used as a correction parameter to gravimetric survey data."""

with open('requirements.txt', 'r') as fd:
    requirements = fd.read().strip().splitlines()

with open('LICENSE', 'r') as fd:
    _license = fd.read().strip()

setup(
    name='LongmanTide',
    version=__version__,
    packages=['longmantide'],
    install_requires=requirements,
    tests_require=['pytest'],
    python_requires='>=3.5.*',
    description="Tide gravitational correction based on I.M. Longman's Formulas for Computing the Tidal Accelerations "
                "Due to the Moon and the Sun",
    long_description=__description__,
    author='Zachery P. Brady, John R. Leeman',
    author_email='bradyzp@dynamicgravitysystems.com',
    url='https://github.com/bradyzp/LongmanTide/',
    download_url='https://github.com/bradyzp/LongmanTide',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: Microsoft',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries',
    ]
)
