# Input
set datafile separator " "

# Output
set terminal postscript eps enhanced color size 2.5,2
set output outfile

load "gnuplot/settings.gnuplot"

# X Axis
set xrange[-1:8]
set xtics ("2" 0, "4" 1, "6" 2, "8" 3, "10" 4, "12" 5, "14" 6, "16" 7)
set xlabel "Number of Clusters"

# Y Axis
set ytic
set xtics

# Grid
set grid xtics
set grid ytics

set macros

set ylabel label

unset key

YAXIS = "set format y yformat; \
		 set yrange [0:]"

LEGEND = "set key inside top right height 1 width 1 box lw 1 font ',12' samplen 1.5"

@YAXIS
@LEGEND
    plot fur using ($2/scale)  with linespoints title "Fur" ls 16, \
        jacobi using ($2/scale) with linespoints title "Jacobi" ls 17, \
        gol using ($2/scale) with linespoints title "GoL" ls 19
