DATADIR=$1
PLOTDIR=$2
label=$3
scale=$4
app=$5
yformat="%1.1f"
#Processing for speedup
# $DATADIR/getSpeedup.sh $DATADIR "Fur"
# $DATADIR/getSpeedup.sh $DATADIR "Gol"
# $DATADIR/getSpeedup.sh $DATADIR "Jacobi"

            gnuplot                                                            \
                -e "label='$label'"                                            \
                -e "yformat='$yformat'"                                        \
                -e "scale='$scale'"                                            \
                -e "appA='$DATADIR/energyTiles32Plot.txt'"                      \
                -e "appB='$DATADIR/energyTiles64Plot.txt'"                      \
                -e "appC='$DATADIR/energyTiles128Plot.txt'"                     \
                -e "outfile='$PLOTDIR/PreviewPlot.eps'"                        \
            gnuplot/MPPAPlotBoxplotEnergy.gnuplot

epstool --copy --bbox $PLOTDIR/PreviewPlot.eps --output $PLOTDIR/MPPAPlot${app}EnergyTiles.eps
epstopdf                     \
    --outfile=$PLOTDIR/MPPAPlot${app}EnergyTiles.pdf  \
    $PLOTDIR/MPPAPlot${app}EnergyTiles.eps

# > ${DATADIR}/tmpTimeValues.txt
rm -r $PLOTDIR/PreviewPlot.eps
rm -r $PLOTDIR/MPPAPlot${app}EnergyTiles.eps
# rm -r $DATADIR/tmpMeanTimeFur.txt
# rm -r $DATADIR/tmpMeanTimeJacobi.txt
# rm -r $DATADIR/tmpMeanTimeGol.txt
