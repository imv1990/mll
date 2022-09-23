//===- VectorDialect.cpp - -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "mll/Dialect/Vector/VectorDialect.h"
#include "mll/AST/AST.h"
#include "mll/Dialect/Vector/VectorDialect.cpp.inc"
#include "mll/Dialect/Vector/VectorDialectNodes.cpp.inc"
#include "mll/Dialect/Vector/VectorDialectTypes.cpp.inc"
#include "mll/Parse/BuiltinDialect.h"
#include "mll/Parse/Parser.h"

namespace mll {
namespace vector {

VectorType *VectorType::parse(mll::Parser &parser, mll::ASTBuilder &builder) {
  if (!parser.parseToken(Token::less)) {
    emitError(parser.getCurrLoc(),
              "Expecting < symbol while parsing array type");
    return nullptr;
  }

  llvm::SmallVector<int64_t> shape;
  while (true) {
    auto intVal = parser.parseInteger();
    if (!intVal.has_value()) {
      emitError(parser.getCurrLoc(), "Expecting integer as dimension size");
      return nullptr;
    }
    shape.push_back(intVal.value());

    if (!parser.parseToken(Token::star)) {
      emitError(parser.getCurrLoc(), "Expecting * between shapes");
      return nullptr;
    }

    if (parser.isToken(Token::identifier)) {
      break;
    }
  }

  auto baseTy = parser.parseType();
  if (!baseTy) {
    return nullptr;
  }

  if (!parser.parseToken(Token::greater)) {
    emitError(parser.getCurrLoc(),
              "Exepecting > symbol while parsing array type");
    return nullptr;
  }

  return builder.getType<VectorType>(shape, baseTy);
}

Expr *parseVectorTypeConstExpr(::mll::Parser &parser,
                               ::mll::ASTBuilder &builder, ::mll::Type *ty) {
  auto arrTy = llvm::cast<VectorType>(ty);
  auto loc = parser.getCurrLoc();
  if (parser.isToken(Token::identifier) && parser.isTokenSpelling("splat")) {
    parser.consumeToken();
    parser.parseToken(Token::l_paren);
    auto expr = parser.parseConstantExpr(arrTy->base());
    assert(expr);
    parser.parseToken(Token::r_paren);
    return builder.create<VectorSplatConstantExpr>(loc, arrTy, expr);
  }

  assert(false && "unhandled vector constant");
  return nullptr;
}

} // namespace vector
} // namespace mll