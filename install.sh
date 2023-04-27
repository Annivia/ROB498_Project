#!/bin/bash

conda create -n ponyz python=3.10 -y
conda init bash
source ~/.bashrc
conda activate ponyz

if ! command -v pip &> /dev/null
then
    conda install pip -y
fi

pip install -r requirements.txt
python demo.py
