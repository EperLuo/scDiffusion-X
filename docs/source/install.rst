.. _install:

Installation
=============

We used conda to deploy the required environment for scMulDiffusion.
::
    conda create --name scmuldiff python=3.8

Prerequisites
-------------
Before installing scMulDiffusion, users should first install Pytorch and other dependancies.
::
    pip install -r requirements.txt

Then, users need to install mpi4py for distribute data parallel training. We recommand using the conda to install:
::
    conda install mpi4py


PyPI
----

scMulDiffusion is available on PyPI here_ and can be installed via::

    pip install scmuldiffusion


Source file
----
Source file of model and scripts can be found in `github <https://github.com/EperLuo/scMulDiffusion/>`_



.. _here: https://pypi.org/project/scmuldiffusion