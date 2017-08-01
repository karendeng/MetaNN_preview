##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Release
ProjectName            :=GeneralTest
ConfigurationName      :=Release
WorkspacePath          :=/home/liwei/MetaNN/MetaNN_new/MetaNN
ProjectPath            :=/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest
IntermediateDirectory  :=./Release
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=liwei
Date                   :=01/08/17
CodeLitePath           :=/home/liwei/.codelite
LinkerName             :=/usr/bin/g++
SharedObjectLinkerName :=/usr/bin/g++ -shared -fPIC
ObjectSuffix           :=.o
DependSuffix           :=.o.d
PreprocessSuffix       :=.i
DebugSwitch            :=-g 
IncludeSwitch          :=-I
LibrarySwitch          :=-l
OutputSwitch           :=-o 
LibraryPathSwitch      :=-L
PreprocessorSwitch     :=-D
SourceSwitch           :=-c 
OutputFile             :=$(IntermediateDirectory)/$(ProjectName)
Preprocessors          :=$(PreprocessorSwitch)NDEBUG 
ObjectSwitch           :=-o 
ArchiveOutputSwitch    := 
PreprocessOnlySwitch   :=-E
ObjectsFileList        :="GeneralTest.txt"
PCHCompileFlags        :=
MakeDirCommand         :=mkdir -p
LinkOptions            :=  
IncludePath            :=  $(IncludeSwitch). $(IncludeSwitch). $(IncludeSwitch)../ 
IncludePCH             := 
RcIncludePath          := 
Libs                   := 
ArLibs                 :=  
LibPath                := $(LibraryPathSwitch). 

##
## Common variables
## AR, CXX, CC, AS, CXXFLAGS and CFLAGS can be overriden using an environment variables
##
AR       := /usr/bin/ar rcu
CXX      := /usr/bin/g++
CC       := /usr/bin/gcc
CXXFLAGS :=  -O2 -Wall -std=c++14 $(Preprocessors)
CFLAGS   :=  -O2 -Wall $(Preprocessors)
ASFLAGS  := 
AS       := /usr/bin/as


##
## User defined environment variables
##
CodeLiteDir:=/usr/share/codelite
Objects0=$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_test_scalar.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_test_general_matrix.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_test_zero_matrix.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_test_trival_matrix.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_test_one_hot_vector.cpp$(ObjectSuffix) $(IntermediateDirectory)/facilities_test_named_params.cpp$(ObjectSuffix) $(IntermediateDirectory)/policies_test_change_policy.cpp$(ObjectSuffix) $(IntermediateDirectory)/policies_test_policy_operations.cpp$(ObjectSuffix) $(IntermediateDirectory)/policies_test_policy_selector.cpp$(ObjectSuffix) \
	$(IntermediateDirectory)/evaluate_test_eval_plan.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_add.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_collapse.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_divide.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_dot.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_element_mul.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_interpolate.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_negative_log_likelihood.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_negative_log_likelihood_derivative.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_sigmoid.cpp$(ObjectSuffix) \
	$(IntermediateDirectory)/operators_test_sigmoid_derivative.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_softmax.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_substract.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_tanh.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_tanh_derivative.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_transpose.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_softmax_derivative.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_abs.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_sign.cpp$(ObjectSuffix) $(IntermediateDirectory)/layers_elementary_test_add_layer.cpp$(ObjectSuffix) \
	$(IntermediateDirectory)/layers_elementary_test_bias_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/layers_elementary_test_element_mul_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/layers_elementary_test_interpolate_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/layers_elementary_test_sigmoid_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/layers_elementary_test_softmax_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/layers_elementary_test_tanh_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/layers_elementary_test_weight_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/layers_elementary_test_abs_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/layers_cost_test_negative_log_likelihood_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/layers_compose_test_compose_kernel.cpp$(ObjectSuffix) \
	$(IntermediateDirectory)/layers_compose_test_linear_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/layers_compose_test_single_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/layers_recurrent_test_gru.cpp$(ObjectSuffix) 



Objects=$(Objects0) 

##
## Main Build Targets 
##
.PHONY: all clean PreBuild PrePreBuild PostBuild MakeIntermediateDirs
all: $(OutputFile)

$(OutputFile): $(IntermediateDirectory)/.d $(Objects) 
	@$(MakeDirCommand) $(@D)
	@echo "" > $(IntermediateDirectory)/.d
	@echo $(Objects0)  > $(ObjectsFileList)
	$(LinkerName) $(OutputSwitch)$(OutputFile) @$(ObjectsFileList) $(LibPath) $(Libs) $(LinkOptions)

MakeIntermediateDirs:
	@test -d ./Release || $(MakeDirCommand) ./Release


$(IntermediateDirectory)/.d:
	@test -d ./Release || $(MakeDirCommand) ./Release

PreBuild:


##
## Objects
##
$(IntermediateDirectory)/main.cpp$(ObjectSuffix): main.cpp $(IntermediateDirectory)/main.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/main.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/main.cpp$(DependSuffix): main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/main.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/main.cpp$(DependSuffix) -MM main.cpp

$(IntermediateDirectory)/main.cpp$(PreprocessSuffix): main.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/main.cpp$(PreprocessSuffix) main.cpp

$(IntermediateDirectory)/data_test_scalar.cpp$(ObjectSuffix): data/test_scalar.cpp $(IntermediateDirectory)/data_test_scalar.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/data/test_scalar.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_test_scalar.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_test_scalar.cpp$(DependSuffix): data/test_scalar.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_test_scalar.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_test_scalar.cpp$(DependSuffix) -MM data/test_scalar.cpp

$(IntermediateDirectory)/data_test_scalar.cpp$(PreprocessSuffix): data/test_scalar.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_test_scalar.cpp$(PreprocessSuffix) data/test_scalar.cpp

$(IntermediateDirectory)/data_test_general_matrix.cpp$(ObjectSuffix): data/test_general_matrix.cpp $(IntermediateDirectory)/data_test_general_matrix.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/data/test_general_matrix.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_test_general_matrix.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_test_general_matrix.cpp$(DependSuffix): data/test_general_matrix.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_test_general_matrix.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_test_general_matrix.cpp$(DependSuffix) -MM data/test_general_matrix.cpp

$(IntermediateDirectory)/data_test_general_matrix.cpp$(PreprocessSuffix): data/test_general_matrix.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_test_general_matrix.cpp$(PreprocessSuffix) data/test_general_matrix.cpp

$(IntermediateDirectory)/data_test_zero_matrix.cpp$(ObjectSuffix): data/test_zero_matrix.cpp $(IntermediateDirectory)/data_test_zero_matrix.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/data/test_zero_matrix.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_test_zero_matrix.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_test_zero_matrix.cpp$(DependSuffix): data/test_zero_matrix.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_test_zero_matrix.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_test_zero_matrix.cpp$(DependSuffix) -MM data/test_zero_matrix.cpp

$(IntermediateDirectory)/data_test_zero_matrix.cpp$(PreprocessSuffix): data/test_zero_matrix.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_test_zero_matrix.cpp$(PreprocessSuffix) data/test_zero_matrix.cpp

$(IntermediateDirectory)/data_test_trival_matrix.cpp$(ObjectSuffix): data/test_trival_matrix.cpp $(IntermediateDirectory)/data_test_trival_matrix.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/data/test_trival_matrix.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_test_trival_matrix.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_test_trival_matrix.cpp$(DependSuffix): data/test_trival_matrix.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_test_trival_matrix.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_test_trival_matrix.cpp$(DependSuffix) -MM data/test_trival_matrix.cpp

$(IntermediateDirectory)/data_test_trival_matrix.cpp$(PreprocessSuffix): data/test_trival_matrix.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_test_trival_matrix.cpp$(PreprocessSuffix) data/test_trival_matrix.cpp

$(IntermediateDirectory)/data_test_one_hot_vector.cpp$(ObjectSuffix): data/test_one_hot_vector.cpp $(IntermediateDirectory)/data_test_one_hot_vector.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/data/test_one_hot_vector.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_test_one_hot_vector.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_test_one_hot_vector.cpp$(DependSuffix): data/test_one_hot_vector.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_test_one_hot_vector.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_test_one_hot_vector.cpp$(DependSuffix) -MM data/test_one_hot_vector.cpp

$(IntermediateDirectory)/data_test_one_hot_vector.cpp$(PreprocessSuffix): data/test_one_hot_vector.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_test_one_hot_vector.cpp$(PreprocessSuffix) data/test_one_hot_vector.cpp

$(IntermediateDirectory)/facilities_test_named_params.cpp$(ObjectSuffix): facilities/test_named_params.cpp $(IntermediateDirectory)/facilities_test_named_params.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/facilities/test_named_params.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/facilities_test_named_params.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/facilities_test_named_params.cpp$(DependSuffix): facilities/test_named_params.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/facilities_test_named_params.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/facilities_test_named_params.cpp$(DependSuffix) -MM facilities/test_named_params.cpp

$(IntermediateDirectory)/facilities_test_named_params.cpp$(PreprocessSuffix): facilities/test_named_params.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/facilities_test_named_params.cpp$(PreprocessSuffix) facilities/test_named_params.cpp

$(IntermediateDirectory)/policies_test_change_policy.cpp$(ObjectSuffix): policies/test_change_policy.cpp $(IntermediateDirectory)/policies_test_change_policy.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/policies/test_change_policy.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/policies_test_change_policy.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/policies_test_change_policy.cpp$(DependSuffix): policies/test_change_policy.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/policies_test_change_policy.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/policies_test_change_policy.cpp$(DependSuffix) -MM policies/test_change_policy.cpp

$(IntermediateDirectory)/policies_test_change_policy.cpp$(PreprocessSuffix): policies/test_change_policy.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/policies_test_change_policy.cpp$(PreprocessSuffix) policies/test_change_policy.cpp

$(IntermediateDirectory)/policies_test_policy_operations.cpp$(ObjectSuffix): policies/test_policy_operations.cpp $(IntermediateDirectory)/policies_test_policy_operations.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/policies/test_policy_operations.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/policies_test_policy_operations.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/policies_test_policy_operations.cpp$(DependSuffix): policies/test_policy_operations.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/policies_test_policy_operations.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/policies_test_policy_operations.cpp$(DependSuffix) -MM policies/test_policy_operations.cpp

$(IntermediateDirectory)/policies_test_policy_operations.cpp$(PreprocessSuffix): policies/test_policy_operations.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/policies_test_policy_operations.cpp$(PreprocessSuffix) policies/test_policy_operations.cpp

$(IntermediateDirectory)/policies_test_policy_selector.cpp$(ObjectSuffix): policies/test_policy_selector.cpp $(IntermediateDirectory)/policies_test_policy_selector.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/policies/test_policy_selector.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/policies_test_policy_selector.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/policies_test_policy_selector.cpp$(DependSuffix): policies/test_policy_selector.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/policies_test_policy_selector.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/policies_test_policy_selector.cpp$(DependSuffix) -MM policies/test_policy_selector.cpp

$(IntermediateDirectory)/policies_test_policy_selector.cpp$(PreprocessSuffix): policies/test_policy_selector.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/policies_test_policy_selector.cpp$(PreprocessSuffix) policies/test_policy_selector.cpp

$(IntermediateDirectory)/evaluate_test_eval_plan.cpp$(ObjectSuffix): evaluate/test_eval_plan.cpp $(IntermediateDirectory)/evaluate_test_eval_plan.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/evaluate/test_eval_plan.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/evaluate_test_eval_plan.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/evaluate_test_eval_plan.cpp$(DependSuffix): evaluate/test_eval_plan.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/evaluate_test_eval_plan.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/evaluate_test_eval_plan.cpp$(DependSuffix) -MM evaluate/test_eval_plan.cpp

$(IntermediateDirectory)/evaluate_test_eval_plan.cpp$(PreprocessSuffix): evaluate/test_eval_plan.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/evaluate_test_eval_plan.cpp$(PreprocessSuffix) evaluate/test_eval_plan.cpp

$(IntermediateDirectory)/operators_test_add.cpp$(ObjectSuffix): operators/test_add.cpp $(IntermediateDirectory)/operators_test_add.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_add.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_add.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_add.cpp$(DependSuffix): operators/test_add.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_add.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_add.cpp$(DependSuffix) -MM operators/test_add.cpp

$(IntermediateDirectory)/operators_test_add.cpp$(PreprocessSuffix): operators/test_add.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_add.cpp$(PreprocessSuffix) operators/test_add.cpp

$(IntermediateDirectory)/operators_test_collapse.cpp$(ObjectSuffix): operators/test_collapse.cpp $(IntermediateDirectory)/operators_test_collapse.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_collapse.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_collapse.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_collapse.cpp$(DependSuffix): operators/test_collapse.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_collapse.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_collapse.cpp$(DependSuffix) -MM operators/test_collapse.cpp

$(IntermediateDirectory)/operators_test_collapse.cpp$(PreprocessSuffix): operators/test_collapse.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_collapse.cpp$(PreprocessSuffix) operators/test_collapse.cpp

$(IntermediateDirectory)/operators_test_divide.cpp$(ObjectSuffix): operators/test_divide.cpp $(IntermediateDirectory)/operators_test_divide.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_divide.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_divide.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_divide.cpp$(DependSuffix): operators/test_divide.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_divide.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_divide.cpp$(DependSuffix) -MM operators/test_divide.cpp

$(IntermediateDirectory)/operators_test_divide.cpp$(PreprocessSuffix): operators/test_divide.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_divide.cpp$(PreprocessSuffix) operators/test_divide.cpp

$(IntermediateDirectory)/operators_test_dot.cpp$(ObjectSuffix): operators/test_dot.cpp $(IntermediateDirectory)/operators_test_dot.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_dot.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_dot.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_dot.cpp$(DependSuffix): operators/test_dot.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_dot.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_dot.cpp$(DependSuffix) -MM operators/test_dot.cpp

$(IntermediateDirectory)/operators_test_dot.cpp$(PreprocessSuffix): operators/test_dot.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_dot.cpp$(PreprocessSuffix) operators/test_dot.cpp

$(IntermediateDirectory)/operators_test_element_mul.cpp$(ObjectSuffix): operators/test_element_mul.cpp $(IntermediateDirectory)/operators_test_element_mul.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_element_mul.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_element_mul.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_element_mul.cpp$(DependSuffix): operators/test_element_mul.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_element_mul.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_element_mul.cpp$(DependSuffix) -MM operators/test_element_mul.cpp

$(IntermediateDirectory)/operators_test_element_mul.cpp$(PreprocessSuffix): operators/test_element_mul.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_element_mul.cpp$(PreprocessSuffix) operators/test_element_mul.cpp

$(IntermediateDirectory)/operators_test_interpolate.cpp$(ObjectSuffix): operators/test_interpolate.cpp $(IntermediateDirectory)/operators_test_interpolate.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_interpolate.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_interpolate.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_interpolate.cpp$(DependSuffix): operators/test_interpolate.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_interpolate.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_interpolate.cpp$(DependSuffix) -MM operators/test_interpolate.cpp

$(IntermediateDirectory)/operators_test_interpolate.cpp$(PreprocessSuffix): operators/test_interpolate.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_interpolate.cpp$(PreprocessSuffix) operators/test_interpolate.cpp

$(IntermediateDirectory)/operators_test_negative_log_likelihood.cpp$(ObjectSuffix): operators/test_negative_log_likelihood.cpp $(IntermediateDirectory)/operators_test_negative_log_likelihood.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_negative_log_likelihood.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_negative_log_likelihood.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_negative_log_likelihood.cpp$(DependSuffix): operators/test_negative_log_likelihood.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_negative_log_likelihood.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_negative_log_likelihood.cpp$(DependSuffix) -MM operators/test_negative_log_likelihood.cpp

$(IntermediateDirectory)/operators_test_negative_log_likelihood.cpp$(PreprocessSuffix): operators/test_negative_log_likelihood.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_negative_log_likelihood.cpp$(PreprocessSuffix) operators/test_negative_log_likelihood.cpp

$(IntermediateDirectory)/operators_test_negative_log_likelihood_derivative.cpp$(ObjectSuffix): operators/test_negative_log_likelihood_derivative.cpp $(IntermediateDirectory)/operators_test_negative_log_likelihood_derivative.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_negative_log_likelihood_derivative.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_negative_log_likelihood_derivative.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_negative_log_likelihood_derivative.cpp$(DependSuffix): operators/test_negative_log_likelihood_derivative.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_negative_log_likelihood_derivative.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_negative_log_likelihood_derivative.cpp$(DependSuffix) -MM operators/test_negative_log_likelihood_derivative.cpp

$(IntermediateDirectory)/operators_test_negative_log_likelihood_derivative.cpp$(PreprocessSuffix): operators/test_negative_log_likelihood_derivative.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_negative_log_likelihood_derivative.cpp$(PreprocessSuffix) operators/test_negative_log_likelihood_derivative.cpp

$(IntermediateDirectory)/operators_test_sigmoid.cpp$(ObjectSuffix): operators/test_sigmoid.cpp $(IntermediateDirectory)/operators_test_sigmoid.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_sigmoid.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_sigmoid.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_sigmoid.cpp$(DependSuffix): operators/test_sigmoid.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_sigmoid.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_sigmoid.cpp$(DependSuffix) -MM operators/test_sigmoid.cpp

$(IntermediateDirectory)/operators_test_sigmoid.cpp$(PreprocessSuffix): operators/test_sigmoid.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_sigmoid.cpp$(PreprocessSuffix) operators/test_sigmoid.cpp

$(IntermediateDirectory)/operators_test_sigmoid_derivative.cpp$(ObjectSuffix): operators/test_sigmoid_derivative.cpp $(IntermediateDirectory)/operators_test_sigmoid_derivative.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_sigmoid_derivative.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_sigmoid_derivative.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_sigmoid_derivative.cpp$(DependSuffix): operators/test_sigmoid_derivative.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_sigmoid_derivative.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_sigmoid_derivative.cpp$(DependSuffix) -MM operators/test_sigmoid_derivative.cpp

$(IntermediateDirectory)/operators_test_sigmoid_derivative.cpp$(PreprocessSuffix): operators/test_sigmoid_derivative.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_sigmoid_derivative.cpp$(PreprocessSuffix) operators/test_sigmoid_derivative.cpp

$(IntermediateDirectory)/operators_test_softmax.cpp$(ObjectSuffix): operators/test_softmax.cpp $(IntermediateDirectory)/operators_test_softmax.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_softmax.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_softmax.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_softmax.cpp$(DependSuffix): operators/test_softmax.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_softmax.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_softmax.cpp$(DependSuffix) -MM operators/test_softmax.cpp

$(IntermediateDirectory)/operators_test_softmax.cpp$(PreprocessSuffix): operators/test_softmax.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_softmax.cpp$(PreprocessSuffix) operators/test_softmax.cpp

$(IntermediateDirectory)/operators_test_substract.cpp$(ObjectSuffix): operators/test_substract.cpp $(IntermediateDirectory)/operators_test_substract.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_substract.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_substract.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_substract.cpp$(DependSuffix): operators/test_substract.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_substract.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_substract.cpp$(DependSuffix) -MM operators/test_substract.cpp

$(IntermediateDirectory)/operators_test_substract.cpp$(PreprocessSuffix): operators/test_substract.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_substract.cpp$(PreprocessSuffix) operators/test_substract.cpp

$(IntermediateDirectory)/operators_test_tanh.cpp$(ObjectSuffix): operators/test_tanh.cpp $(IntermediateDirectory)/operators_test_tanh.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_tanh.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_tanh.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_tanh.cpp$(DependSuffix): operators/test_tanh.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_tanh.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_tanh.cpp$(DependSuffix) -MM operators/test_tanh.cpp

$(IntermediateDirectory)/operators_test_tanh.cpp$(PreprocessSuffix): operators/test_tanh.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_tanh.cpp$(PreprocessSuffix) operators/test_tanh.cpp

$(IntermediateDirectory)/operators_test_tanh_derivative.cpp$(ObjectSuffix): operators/test_tanh_derivative.cpp $(IntermediateDirectory)/operators_test_tanh_derivative.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_tanh_derivative.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_tanh_derivative.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_tanh_derivative.cpp$(DependSuffix): operators/test_tanh_derivative.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_tanh_derivative.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_tanh_derivative.cpp$(DependSuffix) -MM operators/test_tanh_derivative.cpp

$(IntermediateDirectory)/operators_test_tanh_derivative.cpp$(PreprocessSuffix): operators/test_tanh_derivative.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_tanh_derivative.cpp$(PreprocessSuffix) operators/test_tanh_derivative.cpp

$(IntermediateDirectory)/operators_test_transpose.cpp$(ObjectSuffix): operators/test_transpose.cpp $(IntermediateDirectory)/operators_test_transpose.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_transpose.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_transpose.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_transpose.cpp$(DependSuffix): operators/test_transpose.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_transpose.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_transpose.cpp$(DependSuffix) -MM operators/test_transpose.cpp

$(IntermediateDirectory)/operators_test_transpose.cpp$(PreprocessSuffix): operators/test_transpose.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_transpose.cpp$(PreprocessSuffix) operators/test_transpose.cpp

$(IntermediateDirectory)/operators_test_softmax_derivative.cpp$(ObjectSuffix): operators/test_softmax_derivative.cpp $(IntermediateDirectory)/operators_test_softmax_derivative.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_softmax_derivative.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_softmax_derivative.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_softmax_derivative.cpp$(DependSuffix): operators/test_softmax_derivative.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_softmax_derivative.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_softmax_derivative.cpp$(DependSuffix) -MM operators/test_softmax_derivative.cpp

$(IntermediateDirectory)/operators_test_softmax_derivative.cpp$(PreprocessSuffix): operators/test_softmax_derivative.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_softmax_derivative.cpp$(PreprocessSuffix) operators/test_softmax_derivative.cpp

$(IntermediateDirectory)/operators_test_abs.cpp$(ObjectSuffix): operators/test_abs.cpp $(IntermediateDirectory)/operators_test_abs.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_abs.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_abs.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_abs.cpp$(DependSuffix): operators/test_abs.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_abs.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_abs.cpp$(DependSuffix) -MM operators/test_abs.cpp

$(IntermediateDirectory)/operators_test_abs.cpp$(PreprocessSuffix): operators/test_abs.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_abs.cpp$(PreprocessSuffix) operators/test_abs.cpp

$(IntermediateDirectory)/operators_test_sign.cpp$(ObjectSuffix): operators/test_sign.cpp $(IntermediateDirectory)/operators_test_sign.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/operators/test_sign.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_sign.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_sign.cpp$(DependSuffix): operators/test_sign.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_sign.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_sign.cpp$(DependSuffix) -MM operators/test_sign.cpp

$(IntermediateDirectory)/operators_test_sign.cpp$(PreprocessSuffix): operators/test_sign.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_sign.cpp$(PreprocessSuffix) operators/test_sign.cpp

$(IntermediateDirectory)/layers_elementary_test_add_layer.cpp$(ObjectSuffix): layers/elementary/test_add_layer.cpp $(IntermediateDirectory)/layers_elementary_test_add_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/layers/elementary/test_add_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/layers_elementary_test_add_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/layers_elementary_test_add_layer.cpp$(DependSuffix): layers/elementary/test_add_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/layers_elementary_test_add_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/layers_elementary_test_add_layer.cpp$(DependSuffix) -MM layers/elementary/test_add_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_add_layer.cpp$(PreprocessSuffix): layers/elementary/test_add_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/layers_elementary_test_add_layer.cpp$(PreprocessSuffix) layers/elementary/test_add_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_bias_layer.cpp$(ObjectSuffix): layers/elementary/test_bias_layer.cpp $(IntermediateDirectory)/layers_elementary_test_bias_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/layers/elementary/test_bias_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/layers_elementary_test_bias_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/layers_elementary_test_bias_layer.cpp$(DependSuffix): layers/elementary/test_bias_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/layers_elementary_test_bias_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/layers_elementary_test_bias_layer.cpp$(DependSuffix) -MM layers/elementary/test_bias_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_bias_layer.cpp$(PreprocessSuffix): layers/elementary/test_bias_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/layers_elementary_test_bias_layer.cpp$(PreprocessSuffix) layers/elementary/test_bias_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_element_mul_layer.cpp$(ObjectSuffix): layers/elementary/test_element_mul_layer.cpp $(IntermediateDirectory)/layers_elementary_test_element_mul_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/layers/elementary/test_element_mul_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/layers_elementary_test_element_mul_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/layers_elementary_test_element_mul_layer.cpp$(DependSuffix): layers/elementary/test_element_mul_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/layers_elementary_test_element_mul_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/layers_elementary_test_element_mul_layer.cpp$(DependSuffix) -MM layers/elementary/test_element_mul_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_element_mul_layer.cpp$(PreprocessSuffix): layers/elementary/test_element_mul_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/layers_elementary_test_element_mul_layer.cpp$(PreprocessSuffix) layers/elementary/test_element_mul_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_interpolate_layer.cpp$(ObjectSuffix): layers/elementary/test_interpolate_layer.cpp $(IntermediateDirectory)/layers_elementary_test_interpolate_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/layers/elementary/test_interpolate_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/layers_elementary_test_interpolate_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/layers_elementary_test_interpolate_layer.cpp$(DependSuffix): layers/elementary/test_interpolate_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/layers_elementary_test_interpolate_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/layers_elementary_test_interpolate_layer.cpp$(DependSuffix) -MM layers/elementary/test_interpolate_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_interpolate_layer.cpp$(PreprocessSuffix): layers/elementary/test_interpolate_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/layers_elementary_test_interpolate_layer.cpp$(PreprocessSuffix) layers/elementary/test_interpolate_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_sigmoid_layer.cpp$(ObjectSuffix): layers/elementary/test_sigmoid_layer.cpp $(IntermediateDirectory)/layers_elementary_test_sigmoid_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/layers/elementary/test_sigmoid_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/layers_elementary_test_sigmoid_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/layers_elementary_test_sigmoid_layer.cpp$(DependSuffix): layers/elementary/test_sigmoid_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/layers_elementary_test_sigmoid_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/layers_elementary_test_sigmoid_layer.cpp$(DependSuffix) -MM layers/elementary/test_sigmoid_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_sigmoid_layer.cpp$(PreprocessSuffix): layers/elementary/test_sigmoid_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/layers_elementary_test_sigmoid_layer.cpp$(PreprocessSuffix) layers/elementary/test_sigmoid_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_softmax_layer.cpp$(ObjectSuffix): layers/elementary/test_softmax_layer.cpp $(IntermediateDirectory)/layers_elementary_test_softmax_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/layers/elementary/test_softmax_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/layers_elementary_test_softmax_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/layers_elementary_test_softmax_layer.cpp$(DependSuffix): layers/elementary/test_softmax_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/layers_elementary_test_softmax_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/layers_elementary_test_softmax_layer.cpp$(DependSuffix) -MM layers/elementary/test_softmax_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_softmax_layer.cpp$(PreprocessSuffix): layers/elementary/test_softmax_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/layers_elementary_test_softmax_layer.cpp$(PreprocessSuffix) layers/elementary/test_softmax_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_tanh_layer.cpp$(ObjectSuffix): layers/elementary/test_tanh_layer.cpp $(IntermediateDirectory)/layers_elementary_test_tanh_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/layers/elementary/test_tanh_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/layers_elementary_test_tanh_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/layers_elementary_test_tanh_layer.cpp$(DependSuffix): layers/elementary/test_tanh_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/layers_elementary_test_tanh_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/layers_elementary_test_tanh_layer.cpp$(DependSuffix) -MM layers/elementary/test_tanh_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_tanh_layer.cpp$(PreprocessSuffix): layers/elementary/test_tanh_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/layers_elementary_test_tanh_layer.cpp$(PreprocessSuffix) layers/elementary/test_tanh_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_weight_layer.cpp$(ObjectSuffix): layers/elementary/test_weight_layer.cpp $(IntermediateDirectory)/layers_elementary_test_weight_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/layers/elementary/test_weight_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/layers_elementary_test_weight_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/layers_elementary_test_weight_layer.cpp$(DependSuffix): layers/elementary/test_weight_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/layers_elementary_test_weight_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/layers_elementary_test_weight_layer.cpp$(DependSuffix) -MM layers/elementary/test_weight_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_weight_layer.cpp$(PreprocessSuffix): layers/elementary/test_weight_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/layers_elementary_test_weight_layer.cpp$(PreprocessSuffix) layers/elementary/test_weight_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_abs_layer.cpp$(ObjectSuffix): layers/elementary/test_abs_layer.cpp $(IntermediateDirectory)/layers_elementary_test_abs_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/layers/elementary/test_abs_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/layers_elementary_test_abs_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/layers_elementary_test_abs_layer.cpp$(DependSuffix): layers/elementary/test_abs_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/layers_elementary_test_abs_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/layers_elementary_test_abs_layer.cpp$(DependSuffix) -MM layers/elementary/test_abs_layer.cpp

$(IntermediateDirectory)/layers_elementary_test_abs_layer.cpp$(PreprocessSuffix): layers/elementary/test_abs_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/layers_elementary_test_abs_layer.cpp$(PreprocessSuffix) layers/elementary/test_abs_layer.cpp

$(IntermediateDirectory)/layers_cost_test_negative_log_likelihood_layer.cpp$(ObjectSuffix): layers/cost/test_negative_log_likelihood_layer.cpp $(IntermediateDirectory)/layers_cost_test_negative_log_likelihood_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/layers/cost/test_negative_log_likelihood_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/layers_cost_test_negative_log_likelihood_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/layers_cost_test_negative_log_likelihood_layer.cpp$(DependSuffix): layers/cost/test_negative_log_likelihood_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/layers_cost_test_negative_log_likelihood_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/layers_cost_test_negative_log_likelihood_layer.cpp$(DependSuffix) -MM layers/cost/test_negative_log_likelihood_layer.cpp

$(IntermediateDirectory)/layers_cost_test_negative_log_likelihood_layer.cpp$(PreprocessSuffix): layers/cost/test_negative_log_likelihood_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/layers_cost_test_negative_log_likelihood_layer.cpp$(PreprocessSuffix) layers/cost/test_negative_log_likelihood_layer.cpp

$(IntermediateDirectory)/layers_compose_test_compose_kernel.cpp$(ObjectSuffix): layers/compose/test_compose_kernel.cpp $(IntermediateDirectory)/layers_compose_test_compose_kernel.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/layers/compose/test_compose_kernel.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/layers_compose_test_compose_kernel.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/layers_compose_test_compose_kernel.cpp$(DependSuffix): layers/compose/test_compose_kernel.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/layers_compose_test_compose_kernel.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/layers_compose_test_compose_kernel.cpp$(DependSuffix) -MM layers/compose/test_compose_kernel.cpp

$(IntermediateDirectory)/layers_compose_test_compose_kernel.cpp$(PreprocessSuffix): layers/compose/test_compose_kernel.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/layers_compose_test_compose_kernel.cpp$(PreprocessSuffix) layers/compose/test_compose_kernel.cpp

$(IntermediateDirectory)/layers_compose_test_linear_layer.cpp$(ObjectSuffix): layers/compose/test_linear_layer.cpp $(IntermediateDirectory)/layers_compose_test_linear_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/layers/compose/test_linear_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/layers_compose_test_linear_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/layers_compose_test_linear_layer.cpp$(DependSuffix): layers/compose/test_linear_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/layers_compose_test_linear_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/layers_compose_test_linear_layer.cpp$(DependSuffix) -MM layers/compose/test_linear_layer.cpp

$(IntermediateDirectory)/layers_compose_test_linear_layer.cpp$(PreprocessSuffix): layers/compose/test_linear_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/layers_compose_test_linear_layer.cpp$(PreprocessSuffix) layers/compose/test_linear_layer.cpp

$(IntermediateDirectory)/layers_compose_test_single_layer.cpp$(ObjectSuffix): layers/compose/test_single_layer.cpp $(IntermediateDirectory)/layers_compose_test_single_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/layers/compose/test_single_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/layers_compose_test_single_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/layers_compose_test_single_layer.cpp$(DependSuffix): layers/compose/test_single_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/layers_compose_test_single_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/layers_compose_test_single_layer.cpp$(DependSuffix) -MM layers/compose/test_single_layer.cpp

$(IntermediateDirectory)/layers_compose_test_single_layer.cpp$(PreprocessSuffix): layers/compose/test_single_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/layers_compose_test_single_layer.cpp$(PreprocessSuffix) layers/compose/test_single_layer.cpp

$(IntermediateDirectory)/layers_recurrent_test_gru.cpp$(ObjectSuffix): layers/recurrent/test_gru.cpp $(IntermediateDirectory)/layers_recurrent_test_gru.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/MetaNN_new/MetaNN/GeneralTest/layers/recurrent/test_gru.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/layers_recurrent_test_gru.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/layers_recurrent_test_gru.cpp$(DependSuffix): layers/recurrent/test_gru.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/layers_recurrent_test_gru.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/layers_recurrent_test_gru.cpp$(DependSuffix) -MM layers/recurrent/test_gru.cpp

$(IntermediateDirectory)/layers_recurrent_test_gru.cpp$(PreprocessSuffix): layers/recurrent/test_gru.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/layers_recurrent_test_gru.cpp$(PreprocessSuffix) layers/recurrent/test_gru.cpp


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r ./Release/


