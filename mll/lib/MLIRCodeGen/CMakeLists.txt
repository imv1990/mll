set(LIBS
  LLVMSupport
  MLLAST
  MLLParser
  MLIRIR
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRDialect
  MLIRSupport
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
)

add_mlir_library(MLLMLIRCodeGen
  MLIRCodeGen.cpp
  BuiltinDialectConversions.cpp
  DEPENDS
  MLLBuiltinIncGen
  ${LIBS}
)

target_link_libraries(MLLMLIRCodeGen PRIVATE ${LIBS})
llvm_update_compile_flags(MLLMLIRCodeGen)
mlir_check_all_link_libraries(MLLMLIRCodeGen)