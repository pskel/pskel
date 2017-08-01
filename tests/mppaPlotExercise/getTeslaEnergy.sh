#!/bin/bash
SOURCEFILE=$1
DATADIR=$2
APPNAME=$3
grep "rapl:::PACKAGE_ENERGY:PACKAGE0" $1 > ${DATADIR}/tmpSlaveValues.txt
string=$(awk 'FNR==2{ print $2 }' ${DATADIR}/tmpSlaveValues.txt)
echo " $string" &>> ${DATADIR}/tmpTeslaMeanEnergy${APPNAME}.txt
# prev=0
# while read p; do
    # sum=$(($p + $prev))
    # prev=$sum
    # count=$(($count + 1))
# done <${}/tmpMeanTime${APPNAME}.txt
# mean=$(($sum / $count))
# echo "$mean" &>> ${DATADIR}/tmpTime${APPNAME}.txt
