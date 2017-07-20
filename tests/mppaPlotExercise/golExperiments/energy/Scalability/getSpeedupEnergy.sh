#!/bin/bash
DATADIR=$1
CLUSTERBASE=$(head -n 1 ./energyPlotSpeedup.txt)
&> $DATADIR/tmpEnergySpeedup.txt
while read p; do
    # if (( $(bc <<< "$CLUSTERBASE != $p") )); then
    speedup=$(echo "scale=4;$CLUSTERBASE/$p" | bc)
    echo " $speedup" >> $DATADIR/tmpEnergySpeedup.txt
    # fi
done < $DATADIR/energyPlotSpeedup.txt
