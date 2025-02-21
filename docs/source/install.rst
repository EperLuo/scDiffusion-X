.. _install:

Installation
=============

We used conda to deploy the required environment for scDiffusion-X.
::
    conda create --name scmuldiff python=3.8

Prerequisites
-------------
Before installing scDiffusion-X, users should first install Pytorch and other dependancies.
::
    pip install -r requirements.txt

Then, users need to install mpi4py for distribute data parallel training. We recommand using the conda to install:
::
    conda install mpi4py


PyPI
----

scDiffusion-X is available on PyPI here_ and can be installed via::

    pip install scdiffusionX


Source file
----
Source file of model and scripts can be found in `github <https://github.com/EperLuo/scDiffusion-X/>`_



.. _here: https://pypi.org/project/scdiffusionX