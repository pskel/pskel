#!/bin/bash
count=0
echo "$#"
for file in ./*
do
    # number='echo \${$file}'
    echo $file
    # eval echo \${$file}
    # count=$(($count+1))
done
echo "$count"
