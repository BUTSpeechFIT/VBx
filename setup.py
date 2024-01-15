#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

from distutils.core import setup
import glob
import os
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools import find_packages
import tempfile
import zipfile

MODELS_DIR = 'VBx/models'
MODELS = ['ResNet101_8kHz', 'ResNet101_16kHz']


def install_scripts(directory):
    """Call cmd commands to install extra software/repositories.

    Args:
        directory (str): path
    """
    # unpack multiple zip files into .pth file
    for model in MODELS:
        temp_zip = tempfile.NamedTemporaryFile(delete=False)
        nnet_dir = os.path.join(MODELS_DIR, model, 'nnet')
        assert os.path.isdir(nnet_dir), f'{nnet_dir} does not exist.'

        for zip_part in sorted(glob.glob(f'{os.path.join(nnet_dir, "*.pth.zip.part*")}')):
            with open(zip_part, 'rb') as f:
                temp_zip.write(f.read())

        with zipfile.ZipFile(temp_zip, 'r') as fzip:
            fzip.printdir()
            fzip.extractall(path=nnet_dir)


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        self.execute(install_scripts, (self.egg_path,), msg='Running post install scripts')


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        self.execute(install_scripts, (self.install_lib,), msg='Running post install scripts')


setup(
    name='VBx',
    version='1.2',
    packages=find_packages(),
    url='https://github.com/fnlandini/VBx_dev',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'numexpr',
        'fastcluster',
        'h5py',
        'onnxruntime',
        'soundfile',
        'kaldi_io',
        'torch',
        'tabulate',
        'intervaltree'
    ],
    dependency_links=[],
    license='Apache License, Version 2.0',
    cmdclass={'install': PostInstallCommand, 'develop': PostDevelopCommand}
)
