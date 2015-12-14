K1_TOOLCHAIN_DIR ?= /usr/local/k1tools
# set the target platform (hardware or simulator)
platforms := hardware

# cpp-flags
cppflags := -w -I ../../lib/
k1-lflags := -lmppaipc
# slave flags
slave-cflags := -fopenmp -lgomp -lm -I../../lib/ 
slave-lflags := -fopenmp -lgomp -lm -I../../lib/ 

#master
io-bin := master
master-srcs := master.cpp ../../include/interface_mppa.cpp


#slave
cluster-bin := slave
slave-srcs := slave.cpp ../../include/interface_mppa.cpp

# the first binary will be the one booted on the IO!
mppa-bin := multibin
multibin-objs := master slave
multibin-name := pskel.img

# must be at the end of the makefile!
include $(K1_TOOLCHAIN_DIR)/share/make/Makefile.mppaipc
