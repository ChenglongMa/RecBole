#!/bin/bash

echo "Cleaning latex..."
rm -rf ./latex

echo "Cleaning logs..."
rm -rf ./log
rm -rf ./log_tensorboard

echo "Cleaning saved models..."
rm -rf ./saved

echo "Cleaning result..."
rm -rf ./result

echo "Done!"
