#!/bin/sh

rm output/*

for entry in input/*
do
  ./env/bin/python docuscan.py $entry output/"$(basename ${entry})"
done

open output/*



