from setuptools import setup

setup(name='ASE',
      version='0.1',
      description='Bayesian Allelic Specific Expressin using flexible mixture priors',
      url='https://github.com/inti/ASE/',
      author='Inti Pedroso',
      author_email='intipedroso@gmail.com',
      license='MIT',
      packages=['ASE'],
      scripts=['bin/ase.py'],
      zip_safe=False)
