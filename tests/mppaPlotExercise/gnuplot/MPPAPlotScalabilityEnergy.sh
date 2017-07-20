SOURCEPATH=$1
DATADIR=$2
PLOTDIR=$3
label=$4
scale=$5
yformat="%1.1f"

#Processing for speedup
# $DATADIR/getSpeedup.sh $DATADIR "Fur"
# $DATADIR/getSpeedup.sh $DATADIR "Gol"
# $DATADIR/getSpeedup.sh $DATADIR "Jacobi"

            gnuplot                                                            \
                -e "label='$label'"                               \
                -e "yformat='$yformat'"                                        \
                -e "scale='$scale'"                                            \
                -e "fur='$DATADIR/furExperiments/energy/ScalabilityEnergy/energyPlot.txt'"  \
                -e "gol='$DATADIR/golExperiments/energy/Scalability/energyPlot.txt'"  \
                -e "jacobi='$DATADIR/jacobiExperiments/energy/Scalability/energyPlot.txt'"  \
                -e "outfile='$PLOTDIR/PreviewPlot.eps'"                        \
            gnuplot/MPPAPlotScalability.gnuplot

epstool --copy --bbox $PLOTDIR/PreviewPlot.eps --output $PLOTDIR/MPPAPlotScalabilityEnergy.eps
epstopdf                     \
    --outfile=$PLOTDIR/MPPAPlotScalabilityEnergy.pdf  \
    $PLOTDIR/MPPAPlotScalabilityEnergy.eps

# > ${DATADIR}/tmpTimeValues.txt
rm -r $PLOTDIR/PreviewPlot.eps
rm -r $PLOTDIR/MPPAPlotScalabilityEnergy.eps
# rm -r $DATADIR/tmpTimeFur.txt
# rm -r $DATADIR/tmpTimeJacobi.txt
# rm -r $DATADIR/tmpTimeGol.txt
# rm -r $DATADIR/tmpSlaveValues.txt
# rm -r $DATADIR/tmpTimeSpeedupFur.txt
# rm -r $DATADIR/tmpTimeSpeedupJacobi.txt
# rm -r $DATADIR/tmpTimeSpeedupGol.txt
