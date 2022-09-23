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

#ifndef MLL_INCLUDE_PARSER_STATE_H
#define MLL_INCLUDE_PARSER_STATE_H

#include "mll/AST/AST.h"
#include "mll/Parse/Lexer.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SourceMgr.h"

namespace mll {

struct ParserState {
  ParserState(const llvm::SourceMgr &sourceMgr,
              llvm::SmallVector<SymbolTable *, 4> &symbols, MLLContext *context)
      : lex(sourceMgr, context), curToken(lex.lexToken()), symbols(symbols),
        context(context) {}
  ParserState(const ParserState &) = delete;
  void operator=(const ParserState &) = delete;

  Lexer lex;

  Token curToken;

  llvm::SmallVector<SymbolTable *, 4> &symbols;

  MLLContext *context;
};

} // namespace mll
#endif