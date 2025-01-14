# Compiler
NVCC = nvcc

# Directories
SRC_DIR = ./src
OUT_DIR = ./binaryFiles

# Source files
SRC_FILES = $(SRC_DIR)/main.cu $(SRC_DIR)/utilities.cu $(SRC_DIR)/kernels.cu
EXEC = $(OUT_DIR)/main

# Compilation flags
CFLAGS =

# Targets
all: $(EXEC)

# Rule to ensure the output directory exists before compilation
$(EXEC): $(SRC_FILES)
	@mkdir -p $(OUT_DIR) # Ensure output directory exists
	$(NVCC) $(SRC_FILES) -o $@ $(CFLAGS)

# Run the compiled binary
run: $(EXEC)
	$(EXEC)

# Profile using specific metrics
profile: $(EXEC)
	ncu --metrics \
	l1tex__t_bytes.sum.per_second,\
	dram__bytes.sum.per_second,\
	gpu__time_duration.sum,\
	dram__sectors.sum,\
	sm__cycles_active.avg,\
	sm__throughput.avg.pct_of_peak_sustained_elapsed,\
	gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
	smsp__inst_executed.sum,\
	smsp__inst_executed.avg.per_cycle_active,\
	l1tex__t_requests.sum,\
	l1tex__t_sectors_hit.sum \
	$(EXEC)

profileImp:
	ncu --metrics \
	gpu__time_duration.sum,\
	dram__bytes.sum.per_second,\
	smsp__warps_active.avg.pct_of_peak_sustained_elapsed,\
	smsp__inst_executed.avg.per_cycle_active,\
	l1tex__t_sectors_hit.sum,\
	l1tex__t_requests.sum \
	$(EXEC)

profile2:
	nsys profile --trace=cuda --output=program_trace $(EXEC)

profileTime:
	nvprof $(EXEC)

# Clean build artifacts
clean:
	rm -f $(EXEC)

.PHONY: all run profile profileImp profile2 profileTime clean
