//===- OMPDialect.cpp - -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "mll/Dialect/OMP/OMPDialect.h"
#include "mll/AST/AST.h"
#include "mll/Dialect/OMP/OMPDialect.cpp.inc"
#include "mll/Dialect/OMP/OMPDialectNodes.cpp.inc"
#include "mll/Dialect/OMP/OMPDialectTypes.cpp.inc"
#include "mll/Parse/BuiltinDialect.h"
#include "mll/Parse/Parser.h"

namespace mll {
namespace omp {

/// omp.paralllel {
/// <body>
///}
ParallelStmt *ParallelStmt::parse(::mll::Parser &parser,
                                  ::mll::ASTBuilder &builder) {

  auto loc = parser.getCurrLoc();
  auto symTable = builder.create<SymbolTable>("omp-parallel-line-no-" +
                                                  std::to_string(loc.lineNo),
                                              parser.getCurrentSymbolTable());
  auto body = parser.parseBlock(symTable);
  if (!body) {
    return nullptr;
  }

  return builder.create<ParallelStmt>(loc, body, symTable);
}

/// omp.critical {
/// <body>
///}
CriticalStmt *CriticalStmt::parse(::mll::Parser &parser,
                                  ::mll::ASTBuilder &builder) {

  auto loc = parser.getCurrLoc();
  auto symTable = builder.create<SymbolTable>("omp-critical-line-no-" +
                                                  std::to_string(loc.lineNo),
                                              parser.getCurrentSymbolTable());
  auto body = parser.parseBlock(symTable);
  if (!body) {
    return nullptr;
  }

  return builder.create<CriticalStmt>(loc, body, symTable);
}

} // namespace omp
} // namespace mll