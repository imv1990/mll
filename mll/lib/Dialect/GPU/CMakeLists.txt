set(LIBS
  LLVMSupport
  MLLAST
  MLLParser
  MLLMLIRCodeGen
  MLIRGPUOps
)

add_mlir_library(MLLGPUDialect
  GPUDialect.cpp
  GPUDialectConversions.cpp
  DEPENDS
  MLLGPUIncGen
  ${LIBS}
)

target_link_libraries(MLLGPUDialect PRIVATE ${LIBS})
llvm_update_compile_flags(MLLGPUDialect)

mlir_check_all_link_libraries(MLLGPUDialect)