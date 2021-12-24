
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='autonmt',
      version='0.1',
      description='AutoML for Seq2Seq tasks',
      url='https://github.com/salvacarrion/autonmt',
      author='Salva Carrión',
      license='MIT',
      packages=find_packages(),
      package_data={},
      include_package_data=True,
      install_requires=requirements,
      zip_safe=False,
      entry_points={
          'console_scripts': []
      },
      )
