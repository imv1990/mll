//===- ParserState.h - MLL Parser Interface ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MLIR Token.
//
//===----------------------------------------------------------------------===//

#ifndef MLL_INCLUDE_PARSE_TOKEN_H
#define MLL_INCLUDE_PARSE_TOKEN_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

using llvm::SMLoc;
using llvm::SMRange;
using llvm::StringRef;

namespace mll {
class Token {
public:
  enum Kind {
#define TOK_MARKER(NAME) NAME,
#define TOK_IDENTIFIER(NAME) NAME,
#define TOK_LITERAL(NAME) NAME,
#define TOK_PUNCTUATION(NAME, SPELLING) NAME,
#define TOK_KEYWORD(SPELLING) kw_##SPELLING,
#include "mll/Parse/TokenKinds.def"
  };

  Token(Kind kind, StringRef spelling) : kind(kind), spelling(spelling) {}

  // Return the bytes that make up this token.
  StringRef getSpelling() const { return spelling; }

  // Token classification.
  Kind getKind() const { return kind; }
  bool is(Kind k) const { return kind == k; }

  bool isAny(Kind k1, Kind k2) const { return is(k1) || is(k2); }

  /// Return true if this token is one of the specified kinds.
  template <typename... T>
  bool isAny(Kind k1, Kind k2, Kind k3, T... others) const {
    if (is(k1))
      return true;
    return isAny(k2, k3, others...);
  }

  bool isNot(Kind k) const { return kind != k; }

  /// Return true if this token isn't one of the specified kinds.
  template <typename... T> bool isNot(Kind k1, Kind k2, T... others) const {
    return !isAny(k1, k2, others...);
  }

  /// Return true if this is one of the keyword token kinds (e.g. kw_if).
  bool isKeyword() const;

  // Location processing.
  SMLoc getLoc() const;
  SMLoc getEndLoc() const;
  SMRange getLocRange() const;

  int64_t getIntegerValue() const;

  double getFloatingPointValue() const;

  std::string getStringValue() const {
    assert(kind == Token::string);
    auto val = getSpelling().drop_front();
    return val.drop_back().str();
  }

private:
  /// Discriminator that indicates the sort of token this is.
  Kind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;
};
} // namespace mll
#endif