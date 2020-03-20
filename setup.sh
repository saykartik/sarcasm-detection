#!/bin/bash

curr_dir=${PWD##*/}
# Do this only if you havenet already cloned
if [ "$curr_dir" != "sarcasm-detection" ]; then
	git clone https://github.com/saykartik/sarcasm-detection.git
	cd sarcasm-detection
fi

git clone https://github.com/google-research/bert.git bert_repo

pip install emoji
pip install scikit-learn
pip install tensorflow_hub==0.7.0 --force-reinstall
