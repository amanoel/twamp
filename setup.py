from os.path import expanduser
from setuptools import setup

setup(name='twamp',
      version='0.1',
      author='Andre Manoel',
      author_email='andremanoel@gmail.com',
      description='VAMP-like iteration for TV',
      packages=['twamp'],
      install_requires=[
          'numpy',
          'scipy',
          'h5py',
          'scikit-image',
          'matplotlib'
      ])
