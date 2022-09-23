set(LIBS
  LLVMSupport
  MLLAST
  MLLParser
  MLLMLIRCodeGen
  MLIRGPUOps
)

add_mlir_library(MLLOMPDialect
  OMPDialect.cpp
  OMPDialectConversions.cpp
  DEPENDS
  MLLOMPIncGen
  ${LIBS}
)

target_link_libraries(MLLOMPDialect PRIVATE ${LIBS})
llvm_update_compile_flags(MLLOMPDialect)

mlir_check_all_link_libraries(MLLOMPDialect)