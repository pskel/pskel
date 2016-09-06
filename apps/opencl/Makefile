host_ndrange-srcs := host.c

host-bin := host_ndrange

include $(K1_TOOLCHAIN_DIR)/share/make/Makefile.opencl

run_sim: all
	$(K1_TOOLCHAIN_DIR)/bin/k1-pciesim-runner $(BIN_DIR)/host_ndrange

run_hw: all
	$(BIN_DIR)/host_ndrange
