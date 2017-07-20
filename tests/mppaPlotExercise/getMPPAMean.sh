#!/bin/bash
SOURCEFILE=$1
DATADIR=$2
APPNAME=$3
# max=-1
# grep "Slave Time:" ${DATADIR}/$1 > ${DATADIR}/tmpSlaveValues.txt
# while read p; do
    # seded=$(sed -e 's#.*\:\(\)#\1#' <<< "$p")
    # if (( $(echo "$seded > $max" | bc -l) )); then
        # max=$seded
    # fi
# done <${DATADIR}/tmpSlaveValues.txt
# rm -r ${DATADIR}/tmpSlaveValues.txt
# echo "$max" &>> ${DATADIR}/tmpMeanTime${APPNAME}.txt
prev=0
while read p; do
    sum=$(echo $p + $prev | bc)
    prev=$sum
    count=$(($count + 1))
done <${DATADIR}/tmpMeanTime${APPNAME}.txt
mean=$(echo "scale=4;$sum / $count" | bc)
echo " $mean" &>> ${DATADIR}/tmpTime${APPNAME}.txt
