MPPADIR=/usr/local/k1tools

$MPPADIR/bin/k1-jtag-runner --multibinary=output/bin/pskel.img --exec-multibin=IODDR0:master -- $1 $2 $3 $4 $5 $6 $7
