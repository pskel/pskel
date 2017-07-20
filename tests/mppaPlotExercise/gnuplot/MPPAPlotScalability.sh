SOURCEPATH=$1
DATADIR=$2
PLOTDIR=$3
label=$4
scale=$5
yformat="%1.1f"

for spec in 1 2 4 6 8 10 12 14 16
do
    for file in ~/$SOURCEPATH/furExperiments/spec/$spec/*.txt
    do
        $DATADIR/getMPPATime.sh $file $DATADIR "Fur"
    done
    $DATADIR/getMPPAMean.sh $DATADIR/tmpMeanTimeFur.txt $DATADIR "Fur"
    rm -r $DATADIR/tmpMeanTimeFur.txt

    for file in ~/$SOURCEPATH/jacobiExperiments/spec/$spec/*.txt
    do
        $DATADIR/getMPPATime.sh $file $DATADIR "Jacobi"
    done
    $DATADIR/getMPPAMean.sh $DATADIR/tmpMeanTimeJacobi.txt $DATADIR "Jacobi"
    rm -r $DATADIR/tmpMeanTimeJacobi.txt

    for file in ~/$SOURCEPATH/golExperiments/spec/$spec/*.txt
    do
        $DATADIR/getMPPATime.sh $file $DATADIR "Gol"
    done
    $DATADIR/getMPPAMean.sh $DATADIR/tmpMeanTimeGol.txt $DATADIR "Gol"
    rm -r $DATADIR/tmpMeanTimeGol.txt

done
#Processing for speedup
# $DATADIR/getSpeedup.sh $DATADIR "Fur"
# $DATADIR/getSpeedup.sh $DATADIR "Gol"
# $DATADIR/getSpeedup.sh $DATADIR "Jacobi"

            gnuplot                                                            \
                -e "label='Execution Time\(s\)'"                               \
                -e "yformat='$yformat'"                                        \
                -e "scale='$scale'"                                            \
                -e "fur='$DATADIR/tmpTimeFur.txt'"                             \
                -e "jacobi='$DATADIR/tmpTimeJacobi.txt'"                       \
                -e "gol='$DATADIR/tmpTimeGol.txt'"                             \
                -e "outfile='$PLOTDIR/PreviewPlot.eps'"                        \
            gnuplot/MPPAPlotScalability.gnuplot

epstool --copy --bbox $PLOTDIR/PreviewPlot.eps --output $PLOTDIR/MPPAPlotScalability.eps
epstopdf                     \
    --outfile=$PLOTDIR/MPPAPlotScalability.pdf  \
    $PLOTDIR/MPPAPlotScalability.eps

# > ${DATADIR}/tmpTimeValues.txt
rm -r $PLOTDIR/PreviewPlot.eps
rm -r $PLOTDIR/MPPAPlotScalability.eps
rm -r $DATADIR/tmpTimeFur.txt
rm -r $DATADIR/tmpTimeJacobi.txt
rm -r $DATADIR/tmpTimeGol.txt
rm -r $DATADIR/tmpSlaveValues.txt
# rm -r $DATADIR/tmpTimeSpeedupFur.txt
# rm -r $DATADIR/tmpTimeSpeedupJacobi.txt
# rm -r $DATADIR/tmpTimeSpeedupGol.txt
