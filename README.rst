Bayes ASE
===

.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/dfm/emcee/blob/master/LICENSE
.. image:: https://readthedocs.org/projects/bayase/badge/?version=latest
    :target: https://bayase.readthedocs.io/en/latest/?badge=latest

# ASE

Estimate probahility of ASE using Bayesian strategy with prior determined from the data itself using a 
mixture of Beta-Binomial distributions. We assume the allelic imbalance can be modelled by K Beta-Binomial 
distributions for which we fix the mean values, for instance for K = 7 mean values 
are 0.01, 0.16666667, 0.33333333, 0.5, 0.66666667, 0.83333333 and 0.99. 
For each distribution we estimate its variance and prevalence on the data. 
We assume the cental distribution with mean 0.5 corresponde to the best representation of the null distribution 
of no ASE. For each data point we obtain a posterior distribution of by combining the data with the mixture prior 
and then compare this posterior with the null distribution to obtain a probality of not being null or probability of ASE. 


Documentation
-------------

https://bayase.readthedocs.io/en/latest/


# Install
We find easy to get all dependecies using [coda](https://www.anaconda.com/download/) 
Firstly, download the package for instance 
```
git clone https://github.com/inti/ASE.git
cd ASE
```
Create a environment with necessary dependencies

```
conda env create -f environment.yml
```

Install the package, use the pip associated with the conda environmet

```
pip install .
```
