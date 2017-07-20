SOURCEPATH=$1
DATADIR=$2
PLOTDIR=$3
label=$4
scale=$5
app=$6
yformat="%1.1f"
for spec in 32 64 128
do
    for size in 2048 4096 8192 12288
    do
        for file in ~/$SOURCEPATH/${app}Experiments/spec/$size/$spec/*.txt
        do
            $DATADIR/getMPPATime.sh $file $DATADIR "${app}${spec}"
        done
        $DATADIR/getMPPAMean.sh $DATADIR/tmpMeanTime${app}${spec}.txt $DATADIR "${app}${spec}"
        rm -r $DATADIR/tmpMeanTime${app}${spec}.txt

    done
done
#Processing for speedup
# $DATADIR/getSpeedup.sh $DATADIR "Fur"
# $DATADIR/getSpeedup.sh $DATADIR "Gol"
# $DATADIR/getSpeedup.sh $DATADIR "Jacobi"

            gnuplot                                                            \
                -e "label='$label'"                                            \
                -e "yformat='$yformat'"                                        \
                -e "scale='$scale'"                                            \
                -e "appA='$DATADIR/tmpTime${app}32.txt'"                      \
                -e "appB='$DATADIR/tmpTime${app}64.txt'"                      \
                -e "appC='$DATADIR/tmpTime${app}128.txt'"                     \
                -e "outfile='$PLOTDIR/PreviewPlot.eps'"                        \
            gnuplot/MPPAPlotBoxplot.gnuplot

epstool --copy --bbox $PLOTDIR/PreviewPlot.eps --output $PLOTDIR/MPPAPlot${app}TimeTiles.eps
epstopdf                     \
    --outfile=$PLOTDIR/MPPAPlot${app}TimeTiles.pdf  \
    $PLOTDIR/MPPAPlot${app}TimeTiles.eps

# > ${DATADIR}/tmpTimeValues.txt
rm -r $PLOTDIR/PreviewPlot.eps
rm -r $PLOTDIR/MPPAPlot${app}TimeTiles.eps
rm -r $DATADIR/tmpTime${app}32.txt
rm -r $DATADIR/tmpTime${app}64.txt
rm -r $DATADIR/tmpTime${app}128.txt
# rm -r $DATADIR/tmpMeanTimeFur.txt
# rm -r $DATADIR/tmpMeanTimeJacobi.txt
# rm -r $DATADIR/tmpMeanTimeGol.txt
