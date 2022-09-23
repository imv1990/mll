//===- AST.h - MLL AST Interface --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MLIR AST utils.
//
//===----------------------------------------------------------------------===//

#ifndef MLL_INCLUDE_AST_AST_BIN_OP_H
#define MLL_INCLUDE_AST_AST_BIN_OP_H

#include "llvm/ADT/StringRef.h"
namespace mll {

class Dialect;

// FIXME: Add associativity
// Binary operator of custom type
struct BinaryOperator {
  // valid characters in operators are
  // + | - | / | * | [a-zA-Z]+
  std::string op;
  int precedence;
  Dialect *dialect;
  bool isRelational;

  // Ill formed object. Do not use it.
  // FIXME: make it private and relevant classes as friends.
  BinaryOperator()
      : op(""), precedence(-1), dialect(nullptr), isRelational(false) {}

  explicit BinaryOperator(Dialect *d, llvm::StringRef op, bool isRelational,
                          int prec)
      : op(op), precedence(prec), dialect(d), isRelational(isRelational) {}
};

} // namespace mll

#endif