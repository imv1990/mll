get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  LLVMSupport
  MLIRSupport
  MLLParser
  MLIRIR
  MLIRDialect
  MLIRSupport
  MLLMLIRCodeGen
  MLLAST
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRExecutionEngine
  LLVMOrcJIT
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRPass
  MLLGPUDialect
  MLLOMPDialect
  MLLVectorDialect
  MLIROpenMPToLLVMIRTranslation
)

add_compile_definitions(MLL_PRINT_LIB="${CMAKE_BINARY_DIR}/lib/libMLLRuntimePrint.so")
add_compile_definitions(MLIR_PRINT_LIB="${CMAKE_BINARY_DIR}/lib/libmlir_runner_utils.so")
add_compile_definitions(MLIR_C_PRINT_LIB="${CMAKE_BINARY_DIR}/lib/libmlir_c_runner_utils.so")

add_mlir_tool(mll
  mll.cpp

  DEPENDS
  MLLRuntimePrint
  mlir_runner_utils
  mlir_c_runner_utils
  ${LIBS}
)

target_link_libraries(mll PRIVATE ${LIBS})
llvm_update_compile_flags(mll)

mlir_check_all_link_libraries(mll)
