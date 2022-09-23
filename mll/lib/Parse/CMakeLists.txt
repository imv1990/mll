set(LIBS
    LLVMSupport
    MLLAST
  )

add_mlir_library(MLLParser
  Parser.cpp
  Token.cpp
  Lexer.cpp
  BuiltinDialect.cpp
  DEPENDS
  MLLBuiltinIncGen
  ${LIBS}
  )

target_link_libraries(MLLParser PRIVATE ${LIBS})
llvm_update_compile_flags(MLLParser)

mlir_check_all_link_libraries(MLLParser)