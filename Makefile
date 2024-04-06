$(shell mkdir -p results)

# list dataset names (ugly...)
DATASETS = $(notdir $(shell find datasets -maxdepth 1 -mindepth 1 -type d -not -name "_*" -and -not -name ".*"))
# objects
ifeq ($(DATASETS), )
$(error [ERROR] no dataset find!)
endif

OBJS = $(DATASETS:%=results/%_result.jsonl)

# rule for a dataset
results/%_result.jsonl: datasets/%
	python 

# default target
ALL: OBJS

test:
	@echo $(OBJS)

clean:
	-@rm -rf results 

.PHONY: ALL clean