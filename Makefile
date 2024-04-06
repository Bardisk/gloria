$(shell mkdir -p results)

SHELL := bash
DEVICE ?= cuda
BATCHSIZE ?= 128
THRESHOLD ?= 0.55

# list dataset names (ugly...)
DATASETS = $(notdir $(shell find datasets -maxdepth 1 -mindepth 1 -type d -not -name "_*" -and -not -name ".*"))
# objects
ifeq ($(DATASETS), )
$(error [ERROR] no dataset find!)
endif

OBJS = $(DATASETS:%=results/%_result.json)

# rule for a dataset
results/%_result.json: datasets/%
	@python tst.py $* $(DEVICE) $(BATCHSIZE) $(THRESHOLD)

# default target
ALL: $(OBJS)

test:
	@echo $(OBJS)

clean:
	-@rm -rf results 

.PHONY: ALL clean