//===- GPUDialect.cpp - -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "mll/Dialect/GPU/GPUDialect.h"
#include "mll/AST/AST.h"
#include "mll/Dialect/GPU/GPUDialect.cpp.inc"
#include "mll/Dialect/GPU/GPUDialectNodes.cpp.inc"
#include "mll/Dialect/GPU/GPUDialectTypes.cpp.inc"
#include "mll/Parse/BuiltinDialect.h"
#include "mll/Parse/Parser.h"

namespace mll {
namespace gpu {

static builtin::ListExpr *parseListExpr(mll::Parser &parser,
                                        ASTBuilder &builder) {
  if (!parser.parseToken(Token::l_paren)) {
    emitError(parser.getCurrLoc(), "Invalid syntax. Expected '('");
    return nullptr;
  }

  ExprList list;
  while (!parser.isToken(Token::r_paren)) {

    auto expr = parser.parseExpr();
    if (!expr) {
      return nullptr;
    }

    list.push_back(expr);

    if (!parser.isToken((Token::r_paren)) && !parser.parseToken(Token::comma)) {
      emitError(parser.getCurrLoc(), "Expected ',' between expressions");
      return nullptr;
    }
  }

  if (list.empty()) {
    emitError(parser.getCurrLoc(), "Cannot have empty expression list");
    return nullptr;
  }
  parser.parseToken(Token::r_paren);

  auto listExpr = builder.create<builtin::ListExpr>(
      list.front()->getLoc(), builder.getType<builtin::ListType>(), list);

  return listExpr;
}

/// gpu.host_register (arr1, arr2, ..)
HostRegisterStmt *HostRegisterStmt::parse(::mll::Parser &parser,
                                          ::mll::ASTBuilder &builder) {
  auto loc = parser.getCurrLoc();
  if (!parser.parseToken(Token::l_paren)) {
    emitError(parser.getCurrLoc(), "Invalid host_register syntax");
    return nullptr;
  }

  ExprList list;
  while (!parser.isToken(Token::r_paren)) {

    auto expr = parser.parseExpr();
    if (!expr) {
      return nullptr;
    }

    if (!llvm::isa<builtin::ArrayType>(expr->getExprType()) ||
        !llvm::isa<builtin::SymbolExpr>(expr)) {
      emitError(parser.getCurrLoc(),
                "Expecting only array type symbol expressions");
      return nullptr;
    }

    list.push_back(expr);

    if (!parser.isToken((Token::r_paren)) && !parser.parseToken(Token::comma)) {
      emitError(parser.getCurrLoc(), "Expected ',' between expressions");
      return nullptr;
    }
  }

  if (list.empty()) {
    emitError(parser.getCurrLoc(),
              "host_register cannot have empty expression list");
    return nullptr;
  }
  parser.parseToken(Token::r_paren);

  auto listExpr = builder.create<builtin::ListExpr>(
      list.front()->getLoc(), builder.getType<builtin::ListType>(), list);
  return builder.create<HostRegisterStmt>(loc, listExpr);
}

/// gpu.launch blocks=(1, 2, 3) threads = (1, 2, 3) {
///  <body>
///}
LaunchStmt *LaunchStmt::parse(::mll::Parser &parser,
                              ::mll::ASTBuilder &builder) {

  auto loc = parser.getCurrLoc();
  if (!parser.parseKeyword("blocks") || !parser.parseToken(Token::equal)) {
    emitError(parser.getCurrLoc(), "Invalid syntax");
    return nullptr;
  }

  auto blocksExpr = parseListExpr(parser, builder);
  if (!blocksExpr) {
    return nullptr;
  }

  if (blocksExpr->children().size() != 3) {
    emitError(blocksExpr->children().front()->getLoc(),
              "Expected 3 values for grid size");
    return nullptr;
  }

  if (!parser.parseKeyword("threads") || !parser.parseToken(Token::equal)) {
    emitError(parser.getCurrLoc(), "Invalid syntax");
    return nullptr;
  }

  auto threadsExpr = parseListExpr(parser, builder);
  if (!blocksExpr) {
    return nullptr;
  }

  if (threadsExpr->children().size() != 3) {
    emitError(blocksExpr->children().front()->getLoc(),
              "Expected 3 values for grid size");
    return nullptr;
  }

  auto symTable = builder.create<SymbolTable>("gpu-launch-line-no-" +
                                                  std::to_string(loc.lineNo),
                                              parser.getCurrentSymbolTable());

  // Insert GPU block/thread ids.
  llvm::SmallVector<std::string> vals = {"blockIdx",  "blockIdy",  "blockIdz",
                                         "threadIdx", "threadIdy", "threadIdz"};
  for (auto val : vals) {
    auto ivSymbol = builder.create<Symbol>(
        val, builder.getType<builtin::I32Type>(), symTable, true);
    symTable->insertSymbol(ivSymbol);
  }

  auto body = parser.parseBlock(symTable);
  if (!body) {
    return nullptr;
  }

  return builder.create<LaunchStmt>(loc, blocksExpr, threadsExpr, body,
                                    symTable);
}

} // namespace gpu
} // namespace mll