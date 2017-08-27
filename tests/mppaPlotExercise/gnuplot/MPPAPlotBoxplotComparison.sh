SOURCEPATH=$1
DATADIR=$2
PLOTDIR=$3
label=$4
scale=$5
tp=$6
qnt=$7

yformat="%1.1f"
for app in fur gol jacobi
do
    #---------------------------MPPA--------------------------------------
    for file in ~/$SOURCEPATH/${app}Experiments/spec/12288/128/*.txt
    do
        $DATADIR/getMPPATime.sh $file $DATADIR "${app}"
    done
    $DATADIR/getMPPAMean.sh $DATADIR/tmpMeanTime${app}.txt $DATADIR "${app}"
    rm -r $DATADIR/tmpMeanTime${app}.txt
    while read p; do
        echo " $p" &>> tmpTimeMPPA.txt
    done<$DATADIR/tmpTime${app}.txt
    rm -r $DATADIR/tmpTime${app}.txt

    data=~/$SOURCEPATH/${app}Experiments/energy/TileSizeEnergy/energyTiles128Plot.txt
    echo "$(tail -n 1 $data)" &>> tmpEnergyMPPA.txt

    #----------------------------TESLA------------------------------------
    for file in ~/$SOURCEPATH/${app}ExperimentsTesla/spec/12288/${qnt}/*.txt
    do
        $DATADIR/getTeslaTime.sh $file $DATADIR "${app}"
    done
    $DATADIR/getTeslaMean.sh $DATADIR/tmpTeslaMeanTime${app}.txt $DATADIR "${app}"
    rm -r $DATADIR/tmpTeslaMeanTime${app}.txt
    while read p; do
        echo " $p" &>> tmpTeslaTime.txt
    done<$DATADIR/tmpTeslaTime${app}.txt
    rm -r $DATADIR/tmpTeslaTime${app}.txt

    for file in ~/$SOURCEPATH/${app}ExperimentsTesla/spec/12288/${qnt}/*.txt
    do
        $DATADIR/getTeslaEnergy.sh $file $DATADIR "${app}"
        # $DATADIR/getTeslaEnergy.sh $file $DATADIR "${app}"
    done
    $DATADIR/getTeslaMean.sh $DATADIR/tmpTeslaMeanEnergy${app}.txt $DATADIR "${app}"
    rm -r $DATADIR/tmpTeslaMeanEnergy${app}.txt

    while read p; do
        echo " $p" &>> tmpTeslaEnergy.txt
    done<$DATADIR/tmpTeslaTime${app}.txt
    rm -r $DATADIR/tmpTeslaTime${app}.txt
done
#Processing for speedup
# $DATADIR/getSpeedup.sh $DATADIR "Fur"
# $DATADIR/getSpeedup.sh $DATADIR "Gol"
# $DATADIR/getSpeedup.sh $DATADIR "Jacobi"

            gnuplot                                                            \
                -e "label='$label'"                                            \
                -e "yformat='$yformat'"                                        \
                -e "scale='$scale'"                                            \
                -e "dataA='$DATADIR/tmp${tp}MPPA.txt'"                      \
                -e "dataB='$DATADIR/tmpTesla${tp}.txt'"                      \
                -e "outfile='$PLOTDIR/PreviewPlot.eps'"                        \
            gnuplot/MPPAPlotBoxplotHistogram.gnuplot

epstool --copy --bbox $PLOTDIR/PreviewPlot.eps --output $PLOTDIR/Comparison${tp}Tiles${qnt}.eps
epstopdf                     \
    --outfile=$PLOTDIR/Comparison${tp}Tiles${qnt}.pdf  \
    $PLOTDIR/Comparison${tp}Tiles${qnt}.eps

# > ${DATADIR}/tmpTimeValues.txt
rm -r $PLOTDIR/PreviewPlot.eps
rm -r $PLOTDIR/Comparison${tp}Tiles${qnt}.eps
# rm -r $DATADIR/tmpTimeMPPA.txt
# rm -r $DATADIR/tmpEnergyMPPA.txt
# rm -r $DATADIR/tmpTeslaTime.txt
# rm -r $DATADIR/tmpTeslaEnergy.txt
