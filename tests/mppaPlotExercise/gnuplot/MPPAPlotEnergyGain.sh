SOURCEPATH=$1
DATADIR=$2
PLOTDIR=$3
label=$4
scale=$5
yformat="%1.1f"

            gnuplot                                                            \
				-e "label='$label'"                                            \
				-e "yformat='$yformat'"                                        \
				-e "scale='$scale'"                                            \
				-e "fur='$DATADIR/furExperiments/energy/ScalabilityEnergy/tmpEnergySpeedup.txt'" \
                -e "gol='$DATADIR/golExperiments/energy/Scalability/tmpEnergySpeedup.txt'" \
                -e "jacobi='$DATADIR/jacobiExperiments/energy/Scalability/tmpEnergySpeedup.txt'" \
				-e "outfile='$PLOTDIR/PreviewPlot.eps'"                        \
                gnuplot/MPPAPlotSpeedup.gnuplot

epstool --copy --bbox $PLOTDIR/PreviewPlot.eps --output $PLOTDIR/MPPAPlotDecreaseEnergy.eps
epstopdf                     \
    --outfile=$PLOTDIR/MPPAPlotDecreaseEnergy.pdf  \
    $PLOTDIR/MPPAPlotDecreaseEnergy.eps

# > ${DATADIR}/tmpTimeValues.txt
rm -r $PLOTDIR/PreviewPlot.eps
rm -r $PLOTDIR/MPPAPlotDecreaseEnergy.eps
