#!/bin/bash
DATADIR=$1
APPNAME=$2
CLUSTERBASE=$(head -n 1 $DATADIR/tmpTime${APPNAME}.txt)
&> $DATADIR/tmpTimeSpeedup${APPNAME}.txt
while read p; do
    if (( $(bc <<< "$CLUSTERBASE != $p") )); then
        speedup=$(echo "scale=4;$CLUSTERBASE/$p" | bc)
        echo " $speedup" >> $DATADIR/tmpTimeSpeedup${APPNAME}.txt
    fi
done < $DATADIR/tmpTime${APPNAME}.txt
