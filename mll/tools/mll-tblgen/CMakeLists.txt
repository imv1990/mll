set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  TableGen
)

add_tablegen(mll-tblgen MLL
    mll-tblgen.cpp
    DialectGen.cpp
    TypeGen.cpp
    ASTNodesGen.cpp
  )

set_target_properties(mll-tblgen PROPERTIES FOLDER "Tablegenning")
target_link_libraries(mll-tblgen
  PRIVATE
  MLIRSupportIndentedOstream
  MLIRTableGen)

mlir_check_all_link_libraries(mll-tblgen)