K1_TOOLCHAIN_DIR ?= /usr/local/k1tools
# set the target platform (hardware or simulator)
platforms := hardware

# cpp-flags
cppflags := -w -I ../../lib/ #-Wstack-usage=4096
k1-lflags := -lmppaipc
# slave flags
slave-cflags := -fopenmp -lgomp -lm -I../../lib/ 
slave-lflags := -fopenmp -lgomp -lm -I../../lib/ 

#master
io-bin := master
master-srcs := master.cpp interface_mppa.cpp


#slave
cluster-bin := slave
slave-srcs := slave.cpp interface_mppa.cpp

# the first binary will be the one booted on the IO!
mppa-bin := multibin
multibin-objs := master slave
multibin-name := Fur_noAbstraction.img

# must be at the end of the makefile!
include $(K1_TOOLCHAIN_DIR)/share/make/Makefile.mppaipc
