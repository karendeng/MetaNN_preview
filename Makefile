.PHONY: clean All

All:
	@echo "----------Building project:[ MetaNN - Release ]----------"
	@cd "MetaNN" && "$(MAKE)" -f  "MetaNN.mk"
clean:
	@echo "----------Cleaning project:[ MetaNN - Release ]----------"
	@cd "MetaNN" && "$(MAKE)" -f  "MetaNN.mk" clean
