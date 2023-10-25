
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='autonmt',
      version='0.5',
      description='A Framework to atreamline the research of seq2seq models',
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
