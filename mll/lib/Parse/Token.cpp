//===- Token.cpp - MLL  Token Implementation ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Token class for the MLIR textual form.
//
//===----------------------------------------------------------------------===//

#include "mll/Parse/Token.h"

using namespace mll;

SMLoc Token::getLoc() const { return SMLoc::getFromPointer(spelling.data()); }

SMLoc Token::getEndLoc() const {
  return SMLoc::getFromPointer(spelling.data() + spelling.size());
}

SMRange Token::getLocRange() const { return SMRange(getLoc(), getEndLoc()); }

int64_t Token::getIntegerValue() const {
  assert(getKind() == integer);
  int64_t Result = 0;
  if (spelling.getAsInteger(10, Result)) {
    assert(false && "failed to get integer");
  }
  return Result;
}

/// For a floatliteral, return its value as a double. Return None if the value
/// underflows or overflows.
double Token::getFloatingPointValue() const {
  double result = 0;
  if (spelling.getAsDouble(result))
    assert(false && "failed to get float");
  return result;
}

/// Return true if this is one of the keyword token kinds (e.g. kw_if).
bool Token::isKeyword() const {
  switch (kind) {
  default:
    return false;
#define TOK_KEYWORD(SPELLING)                                                  \
  case kw_##SPELLING:                                                          \
    return true;
#include "mll/Parse/TokenKinds.def"
  }
}