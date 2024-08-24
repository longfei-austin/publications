default: all

# host compiler
HOST_COMPILER ?= mpic++ -fsycl -fsycl-targets=spir64_gen -fdebug-info-for-profiling # -gline-tables-only # icpx # g++

NAMESPACE_DIR := ../src/cpp

CPP_DIR_SRC := ../src/cpp
CPP_DIR_OBJ := ../obj/cpp
CPP_DIR_DEP := ../obj/cpp

CPP_SRCS_LOCAL := $(wildcard $(CPP_DIR_SRC)/*.cpp)
CPP_OBJS_LOCAL := $(patsubst $(CPP_DIR_SRC)/%.cpp,$(CPP_DIR_OBJ)/%.o,$(CPP_SRCS_LOCAL))
CPP_DEPS_LOCAL := $(patsubst $(CPP_DIR_SRC)/%.cpp,$(CPP_DIR_DEP)/%.d,$(CPP_SRCS_LOCAL))

CPP_DIR_SRC_MAIN := ../src/main_cpp
CPP_DIR_OBJ_MAIN := ../obj/main_cpp
CPP_DIR_DEP_MAIN := ../obj/main_cpp

CPP_SRCS_LOCAL_MAIN := $(wildcard $(CPP_DIR_SRC_MAIN)/*.cpp)
CPP_OBJS_LOCAL_MAIN := $(patsubst $(CPP_DIR_SRC_MAIN)/%.cpp,$(CPP_DIR_OBJ_MAIN)/%.o,$(CPP_SRCS_LOCAL_MAIN))
CPP_DEPS_LOCAL_MAIN := $(patsubst $(CPP_DIR_SRC_MAIN)/%.cpp,$(CPP_DIR_DEP_MAIN)/%.d,$(CPP_SRCS_LOCAL_MAIN))

INC_FLAGS  =
INC_FLAGS_HOST = -I$(CPP_DIR_SRC) -I$(CPP_DIR_SRC_MAIN) $(INC_FLAGS) -I$(NAMESPACE_DIR)

CPPFLAGS_HOST = -O3 -Wall -fmessage-length=0 -std=c++20 -march=native -fp-model precise -no-ftz #-fdebug-info-for-profiling -gline-tables-only # -Wconversion -Warith-conversion
CPPFLAGS_HOST += $(INC_FLAGS_HOST)

MACROFLAGS =

DPPFLAGS = -MMD -MP -MF "$(@:%.o=%.d)" -MT "$(@)"
# NOTE: For some reason, we need a space after "-MF" and "-MT" for it to work with 
# 		nvcc, while with gcc it is fine without space.

LINKER_FLAGS = -lstdc++fs -Xsycl-target-backend "-device pvc"
# [2022/12/10] 
# NOTE: This linker flag -lstdc++fs is needed for some older version of the compiler if 
#		we need to #include <experimental/filesystem> instead of #include <filesystem>. 

# Additional compiler flag
CPPFLAGS_HOST += # -g  
# NOTE: -g is needed for using gdb


CPP_EXE_DIR := .
CPP_EXE_LIST := $(patsubst $(CPP_DIR_SRC_MAIN)/%.cpp,$(CPP_EXE_DIR)/%.exe,$(CPP_SRCS_LOCAL_MAIN))


all : $(CPP_EXE_LIST)
# 	@echo 'executables: $(CPP_EXE_LIST)'
# 	@echo 'object files: $(CPP_OBJS_LOCAL_MAIN)'


CPP_EXE_NAME := $(patsubst $(CPP_DIR_SRC_MAIN)/%.cpp,$(CPP_EXE_DIR)/%,$(CPP_SRCS_LOCAL_MAIN))
$(CPP_EXE_NAME) : $(CPP_EXE_DIR)/% : $(CPP_EXE_DIR)/%.exe


# [2022/12/10]
# NOTE: When linking against (static) libraries, be sure to place the (user compiled) 
# 		object files first.

# Linking
$(CPP_EXE_LIST) : $(CPP_EXE_DIR)/%.exe : $(CPP_DIR_OBJ_MAIN)/%.o $(CPP_OBJS_LOCAL) 
	@echo 'Invoking linker for target: $@'
	@echo 'main object to be linked: $<'
	@echo ' '
	$(HOST_COMPILER) -o $@ $^ $(LINKER_FLAGS)
	@echo 'Finished building target: $@'
	@echo ' '

# compiling .cpp
$(CPP_OBJS_LOCAL_MAIN) : $(CPP_DIR_OBJ_MAIN)/%.o: $(CPP_DIR_SRC_MAIN)/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking gcc Compiler'
	$(HOST_COMPILER) $(CPPFLAGS_HOST) $(DPPFLAGS) $(MACROFLAGS) -o $@ -c $<
	@echo 'Finished building: $<'
	@echo ' '

$(CPP_DIR_OBJ)/%.o: $(CPP_DIR_SRC)/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking gcc Compiler'
	${HOST_COMPILER} $(CPPFLAGS_HOST) $(DPPFLAGS) $(MACROFLAGS) -o "$@" -c "$<"
	@echo 'Finished building: $<'
	@echo ' '


-include $(CPP_OBJS_LOCAL:.o=.d) $(CPP_OBJS_LOCAL_MAIN:.o=.d)

# Other Targets
RM := rm -rf

clm : clean_main
# [2023/01/31] NOTE: clm is shorthand for clean_main

clean_main::
	-$(RM) $(CPP_DEPS_LOCAL_MAIN) $(CPP_OBJS_LOCAL_MAIN) $(CPP_EXE_LIST)
	-@echo ' '

clean:
	-$(RM) $(CPP_DEPS_LOCAL) $(CPP_OBJS_LOCAL) \
		   $(CPP_DEPS_LOCAL_MAIN) $(CPP_OBJS_LOCAL_MAIN) \
		   $(CPP_EXE_LIST)
	-@echo ' '

.PHONY: all clean dependents
