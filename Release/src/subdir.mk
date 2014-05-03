################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/raycastingGL.cpp 

CU_SRCS += \
../src/raycasting.cu 

CU_DEPS += \
./src/raycasting.d 

OBJS += \
./src/raycasting.o \
./src/raycastingGL.o 

CPP_DEPS += \
./src/raycastingGL.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/bin/nvcc -I"/opt/cuda/samples/3_Imaging" -I"/opt/cuda/samples/common/inc" -I"/home/guru/cuda-workspace/CudaRaycaster" -O3 -gencode arch=compute_12,code=sm_12  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/bin/nvcc --compile -I"/opt/cuda/samples/3_Imaging" -I"/opt/cuda/samples/common/inc" -I"/home/guru/cuda-workspace/CudaRaycaster" -O3 -gencode arch=compute_12,code=compute_12 -gencode arch=compute_12,code=sm_12  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/bin/nvcc -I"/opt/cuda/samples/3_Imaging" -I"/opt/cuda/samples/common/inc" -I"/home/guru/cuda-workspace/CudaRaycaster" -O3 -gencode arch=compute_12,code=sm_12  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/bin/nvcc -I"/opt/cuda/samples/3_Imaging" -I"/opt/cuda/samples/common/inc" -I"/home/guru/cuda-workspace/CudaRaycaster" -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

