#!/bin/bash
SOURCEFILE=$1
DATADIR=$2
APPNAME=$3
max=-1
grep "CPU_time" $1 > ${DATADIR}/tmpSlaveValues.txt
while read p; do
    # seded=$(sed -e 's#.*\ \(\)#\1#' <<< "$p")
    seded=$(awk '{ print $2 }' <<< "$p")
done<${DATADIR}/tmpSlaveValues.txt
echo " $seded" &>> ${DATADIR}/tmpTeslaMeanTime${APPNAME}.txt
# prev=0
# while read p; do
    # sum=$(($p + $prev))
    # prev=$sum
    # count=$(($count + 1))
# done <${}/tmpMeanTime${APPNAME}.txt
# mean=$(($sum / $count))
# echo "$mean" &>> ${DATADIR}/tmpTime${APPNAME}.txt
