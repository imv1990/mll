//===- ParserState.h - MLL Parser Interface ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MLIR PraserState.
//
//===----------------------------------------------------------------------===//

#ifndef MLL_INCLUDE_LEXER_STATE_H
#define MLL_INCLUDE_LEXER_STATE_H

#include "mll/AST/AST.h"
#include "mll/Parse/Token.h"
#include "llvm/Support/SourceMgr.h"

namespace mll {

class Lexer {
public:
  explicit Lexer(const llvm::SourceMgr &sourceMgr, MLLContext *context);

  Token lexToken();

  const llvm::SourceMgr &getSourceMgr() { return sourceMgr; }

  /// Encode the specified source location information into a Location object
  /// for attachment to the IR or error reporting.
  Location getEncodedSourceLocation(llvm::SMLoc loc);

  StringRef getBufferIdentifier() const {
    unsigned mainFileID = sourceMgr.getMainFileID();
    auto *buffer = sourceMgr.getMemoryBuffer(mainFileID);
    return buffer->getBufferIdentifier();
  }

private:
  // Helpers.
  Token formToken(Token::Kind kind, const char *tokStart) {
    return Token(kind, StringRef(tokStart, curPtr - tokStart));
  }

  Token emitError(const char *loc, const llvm::Twine &message);

  // Lexer implementation methods.
  Token lexBareIdentifierOrKeyword(const char *tokStart);
  Token lexNumber(const char *tokStart);
  Token lexString(const char *tokStart);

  /// Skip a comment line, starting with a '//'.
  void skipComment();

  const llvm::SourceMgr &sourceMgr;
  MLLContext *context;

  llvm::StringRef curBuffer;
  const char *curPtr;

  Lexer(const Lexer &) = delete;
  void operator=(const Lexer &) = delete;
};
} // namespace mll
#endif