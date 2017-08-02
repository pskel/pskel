set encoding utf8
# Input
set datafile separator " "
set title app
# Output
set terminal postscript eps enhanced color size 2.5,2
set output outfile

load "gnuplot/settings.gnuplot"

# X Axis
set xtics ('2048²' 0, '4096²' 1, '8192²' 2, '12288²' 3)

# Y Axis
set ytic
set xtics

# Grid
set grid xtics
set grid ytics

set macros

set ylabel label


YAXIS = "set format y yformat; \
		 set yrange [0:]"

LEGEND = "set key inside top left height 1 width 1 box lw 1 font ',12' title 'Tamanho do Tile' samplen 1.5"

@YAXIS
@LEGEND
        # set style fill solid 0.8 border -1
        # plot fur using ($2/scale)  with boxes title "Fur" ls 16, \
	# set key inside top right height 0.5 title "Num. Chunks {/Symbol \243} 768"
	plot appA using ($2/scale) with boxes title "32x32" fs pattern 0 lt -1, \
        appB using ($2/scale) with boxes title "64x64" fs pattern 2 lt -1, \
        appC using ($2/scale) with boxes title "128x128" fs pattern 4 lt -1
        # jacobi using ($2/scale) with boxes title "Jacobi" ls 17, \
        # gol using ($2/scale) with boxes title "GoL" ls 19
unset multiplot
