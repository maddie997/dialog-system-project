#!/usr/bin/env bash

# download punkt first
python download_punkt.py

echo "Creating Train Data"
python create_ubuntu_dataset.py "$@" --output 'train.csv' 'train'
#echo "Creating Test Data"
#python create_ubuntu_dataset.py "$@" --output 'test.csv' 'test'
#echo "Creating Valid Data"
#python create_ubuntu_dataset.py "$@" --output 'valid.csv' 'valid'
