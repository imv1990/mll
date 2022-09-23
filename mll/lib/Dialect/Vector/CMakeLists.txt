set(LIBS
  LLVMSupport
  MLLAST
  MLLParser
  MLLMLIRCodeGen
  MLIRGPUOps
)

add_mlir_library(MLLVectorDialect
  VectorDialect.cpp
  VectorDialectConversions.cpp
  DEPENDS
  MLLVectorIncGen
  ${LIBS}
)

target_link_libraries(MLLVectorDialect PRIVATE ${LIBS})
llvm_update_compile_flags(MLLVectorDialect)

mlir_check_all_link_libraries(MLLVectorDialect)