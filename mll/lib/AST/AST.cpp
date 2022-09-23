//===- BuiltinDialect.cpp - -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "mll/AST/AST.h"
#include "mlir/Support/IndentedOstream.h"
#include "mll/AST/Type.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

using namespace mll;

void MLLContext::registerBinaryOperator(BinaryOperator op) {
  if (binaryOpMap.find(op.op) != binaryOpMap.end()) {
    llvm::report_fatal_error("Binary operator ` " + op.op +
                             " ` is being registered again in dialect " +
                             op.dialect->getName());
  }
  binaryOpMap[op.op] = op;
}

void Symbol::dump(llvm::raw_ostream &os) const {
  mlir::raw_indented_ostream fos(os);
  fos << "{\n";
  fos << "Symbol: " << name << "\n";
  auto &ios = fos.indent();
  ios << "Type: " << *type << "\n";
  ios << "SymbolTable: " << parent->getName() << "\n";
  fos.unindent();
  fos << "}\n";
}

void SymbolTable::dump(llvm::raw_ostream &os) const {
  mlir::raw_indented_ostream fos(os);
  fos << "{\n";
  fos << "SymbolTable: " << name << "\n";
  auto &ios = fos.indent();
  ios << "Names: [";
  for (auto &sym : variableMap) {
    ios << sym.first << ", ";
  }
  ios << "]\n";
  fos.unindent();
  fos << "}\n";
}

void Block::dump(llvm::raw_ostream &os) const {

  mlir::raw_indented_ostream ros(os);
  ros << "{\n";
  auto &fos = ros.indent();
  for (auto stmt : nodes) {
    fos << *stmt;
  }
  fos.unindent();
  ros << "}\n";
}