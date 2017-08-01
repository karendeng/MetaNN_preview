.PHONY: clean All

All:
	@echo "----------Building project:[ GeneralTest - Release ]----------"
	@cd "GeneralTest" && "$(MAKE)" -f  "GeneralTest.mk"
clean:
	@echo "----------Cleaning project:[ GeneralTest - Release ]----------"
	@cd "GeneralTest" && "$(MAKE)" -f  "GeneralTest.mk" clean
