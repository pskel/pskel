MPPADIR=/usr/local/k1tools

$MPPADIR/bin/k1-jtag-runner --multibinary=output/bin/pskel.img --exec-multibin=IODDR0:master 0$1 0$2 0$3 0$4 0$5 0$6
