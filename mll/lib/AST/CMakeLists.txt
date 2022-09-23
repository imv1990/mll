set(LIBS
  LLVMSupport
)

add_mlir_library(MLLAST
  AST.cpp
  DEPENDS
  ${LIBS}
)

target_link_libraries(MLLAST PRIVATE ${LIBS})
llvm_update_compile_flags(MLLAST)

mlir_check_all_link_libraries(MLLAST)