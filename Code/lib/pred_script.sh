#!/bin/bash
for i in {06..30}
do
	python -W ignore predict.py --dir_out /home/ivank/dbmount/MozzWearPlot/2021-11-$i/ --to_dash True /home/ivank/dbmount/MozzWear/2021-11-$i/ .aac
done
