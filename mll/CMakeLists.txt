cmake_minimum_required(VERSION 3.12)

project(mll LANGUAGES CXX C)

include(AddLLVM)


# MLIR related
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include) # --src-root
set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include) # --includedir
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)
set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

# MLL related
set(MLL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(MLL_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mll/include)
include_directories(${MLL_TABLEGEN_OUTPUT_DIR})

set(MLL_TABLEGEN_EXE $<TARGET_FILE:mll-tblgen>)

function(mll_tablegen ofn)
    tablegen(MLL ${ARGV})
    set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
        PARENT_SCOPE)
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
endfunction()

function(add_mll_dialect dialect name)
    set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
    mll_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect-name ${name})
    mll_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect-name ${name})
    mll_tablegen(${dialect}DialectTypes.h.inc -gen-type-decls -dialect-name-ty ${name})
    mll_tablegen(${dialect}DialectTypes.cpp.inc -gen-type-defs -dialect-name-ty ${name})
    mll_tablegen(${dialect}DialectNodes.h.inc -gen-ast-node-decls -dialect-name-node ${name})
    mll_tablegen(${dialect}DialectNodes.cpp.inc -gen-ast-node-defs -dialect-name-node ${name})
    add_public_tablegen_target(MLL${dialect}IncGen)

    # Append the includes used for this file to the tablegen_compile_commands
    # file.
    file(APPEND ${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml
        "--- !FileInfo:\n"
        "  filepath: \"${LLVM_TARGET_DEFINITIONS_ABSOLUTE}\"\n"
        "  includes: \"${CMAKE_CURRENT_SOURCE_DIR};${tblgen_includes}\"\n"
    )
endfunction()

include_directories(include/)

add_subdirectory(runtime)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
