from setuptools import setup, find_packages
import pathlib
import os

root = pathlib.Path(__file__).parent
os.chdir(str(root))

setup(name='gym_carla',
      version='0.1.0',
      author='Rudy Garcia A.',
      author_email='c01010010@hotmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['gym', 'pygame']
)
