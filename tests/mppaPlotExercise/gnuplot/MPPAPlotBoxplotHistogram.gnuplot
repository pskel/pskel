set encoding utf8
# Input
set datafile separator " "

# Output
set terminal postscript eps enhanced color size 2.5,2
set output outfile

load "gnuplot/settings.gnuplot"

# X Axis
set xtics ('Fur' 0, 'GoL' 1, 'Jacobi' 2)
#rotate by 30 right nomirror

# Y Axis
set ytic
set xtics center offset 1, 0
# set xtics

# Grid
set grid xtics
set grid ytics

set macros

set ylabel label

YAXIS = "set format y yformat; \
		 set yrange [0:]"

LEGEND = "set key inside top right height 1 width 1 box lw 1 font ',12' samplen 1.5"

@YAXIS
@LEGEND
	plot dataA using ($2/scale) with histogram title "MPPA" fs pattern 0 lt -1, \
        dataB using ($2/scale) with histogram title "Intel" fs pattern 2 lt -1
unset multiplot
