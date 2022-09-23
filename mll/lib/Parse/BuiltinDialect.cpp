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

#include "mll/Parse/BuiltinDialect.h"

#include "mll/AST/AST.h"
#include "mll/Parse/BuiltinDialect.cpp.inc"

#include "mll/Parse/BuiltinDialectTypes.cpp.inc"

#include "mll/Parse/BuiltinDialectNodes.cpp.inc"

#include "mll/Parse/Parser.h"

MLIR_DEFINE_EXPLICIT_TYPE_ID(mll::builtin::ListExpr)

namespace mll {
namespace builtin {
ListExpr::ListExpr(Location loc, builtin::ListType *ty, ExprListRef nodes)
    : Expr(loc, TypeID::get<ListExpr>(), std::make_pair("builtin", "list_expr"),
           ty, nodes, llvm::None, Expr::GenericExprKind) {}

void ListExpr::dump(::llvm::raw_ostream &os) const {
  mlir::raw_indented_ostream ros(os);
  auto &fos = ros.indent();
  os << "builtin.ListExpr: "
     << " {\n";
  fos << "[\n";
  for (auto node : children()) {
    fos << *node << ",\n";
  }
  fos << "]\n";
  fos.unindent();
  os << "}";
}

PrintStmt *PrintStmt::parse(mll::Parser &parser, mll::ASTBuilder &builder) {

  auto loc = parser.getCurrLoc();
  if (!parser.parseToken(Token::l_paren)) {
    emitError(parser.getCurrLoc(), "Invalid print syntax statement");
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
    emitError(parser.getCurrLoc(), "Print cannot have expression list");
    return nullptr;
  }
  parser.parseToken(Token::r_paren);

  auto listExpr = builder.create<ListExpr>(list.front()->getLoc(),
                                           builder.getType<ListType>(), list);
  return builder.create<PrintStmt>(loc, listExpr);
}

ArrayType *ArrayType::parse(mll::Parser &parser, mll::ASTBuilder &builder) {
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

  return builder.getType<ArrayType>(shape, baseTy);
}

Expr *parseI32TypeConstExpr(mll::Parser &parser, mll::ASTBuilder &builder,
                            mll::Type *ty) {

  auto intVal = parser.parseInteger();
  if (!intVal.has_value()) {
    emitError(parser.getCurrLoc(), "Expecing simple integer literal");
    return nullptr;
  }

  return builder.create<I32ConstantExpr>(
      parser.getCurrLoc(), llvm::cast<I32Type>(ty), intVal.value());
}

Expr *parseF32TypeConstExpr(mll::Parser &parser, mll::ASTBuilder &builder,
                            mll::Type *ty) {
  auto intVal = parser.parseFloat();
  if (!intVal.has_value()) {
    emitError(parser.getCurrLoc(), "Expecing simple float literal");
    return nullptr;
  }

  return builder.create<F32ConstantExpr>(
      parser.getCurrLoc(), llvm::cast<F32Type>(ty), intVal.value());
  return nullptr;
}

Expr *parseStringTypeConstExpr(mll::Parser &parser, mll::ASTBuilder &builder,
                               mll::Type *ty) {
  return nullptr;
}

Expr *parseArrayTypeConstExpr(::mll::Parser &parser, ::mll::ASTBuilder &builder,
                              ::mll::Type *ty) {
  auto arrTy = llvm::cast<ArrayType>(ty);
  auto loc = parser.getCurrLoc();
  if (parser.isToken(Token::identifier) && parser.isTokenSpelling("dense")) {
    parser.consumeToken();
    parser.parseToken(Token::l_paren);
    auto expr = parser.parseConstantExpr(arrTy->base());
    assert(expr);
    parser.parseToken(Token::r_paren);
    return builder.create<DenseArrayConstantExpr>(loc, arrTy, expr);
  }

  assert(false && "unhandled array constant");
  return nullptr;
}

Expr *parseArrayTypeVariableExpr(::mll::Parser &parser,
                                 ::mll::ASTBuilder &builder, ::mll::Type *ty) {

  auto var = parser.getTokenSpelling();
  auto loc = parser.getCurrLoc();
  auto arrTy = llvm::cast<ArrayType>(ty);
  auto symbol = parser.getSymbol(var);
  assert(symbol);

  // Identifier.
  parser.consumeToken();

  // Handle array access expression.
  if (parser.isToken(Token::l_square)) {
    parser.consumeToken();

    ExprList list;
    while (!parser.isToken(Token::r_square)) {
      auto expr = parser.parseExpr();
      if (!expr) {
        return nullptr;
      }
      list.push_back(expr);
      parser.parseToken(Token::comma);
    }

    auto val = parser.parseToken(Token::r_square);
    assert(val);

    if (list.size() != arrTy->shape().size()) {
      emitError(loc, "Invalid number of indices for array access. Expecting "
                     "number of dims to be ")
          << arrTy->shape().size();
      return nullptr;
    }

    auto listExpr =
        builder.create<ListExpr>(loc, builder.getType<ListType>(), list);

    return builder.create<ArrayAccessVariableExpr>(loc, arrTy->base(), symbol,
                                                   listExpr);
  }

  return builder.create<builtin::SymbolExpr>(parser.getCurrLoc(), ty, symbol);
}

ForStmt *ForStmt::parse(mll::Parser &parser, mll::ASTBuilder &builder) {
  auto loc = parser.getCurrLoc();

  auto varVal = parser.parseIdentifier();
  if (!varVal.has_value()) {
    emitError(parser.getCurrLoc(),
              "Expecting simple induction variable name for `for` statement");
    return nullptr;
  }

  std::string iv = varVal.value();
  if (parser.getSymbol(iv) != nullptr) {
    emitError(loc, "Cannot use existing symbol as loop induction variable");
    return nullptr;
  }

  if (!parser.parseKeyword("in") || !parser.parseKeyword("range")) {
    emitError(parser.getCurrLoc(), "Error in for loop syntax");
    return nullptr;
  }

  if (!parser.parseToken(Token::l_paren)) {
    emitError(parser.getCurrLoc(), "Invalid `for` loop syntax");
    return nullptr;
  }

  // Parse expression list upto 3 values.
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

  assert(parser.parseToken(Token::r_paren));

  if (list.empty() || list.size() > 3) {
    emitError(parser.getCurrLoc(),
              "Expected range operands (lb, ub, step) or (lb, ub) or (ub)");
    return nullptr;
  }

  for (auto &val : list) {
    if (!llvm::isa<I32Type>(val->getExprType())) {
      emitError(val->getLoc(),
                "Only i32 types are expected for loop bounds and step");
      return nullptr;
    }
  }

  Expr *lb, *ub, *step;

  auto getI32Value = [&](int val) {
    return builder.create<I32ConstantExpr>(parser.getCurrLoc(),
                                           builder.getType<I32Type>(), val);
  };

  if (list.size() == 3) {
    lb = list[0];
    ub = list[1];
    step = list[2];
  } else if (list.size() == 2) {
    lb = list[0];
    ub = list[1];
    step = getI32Value(1);
  } else {
    lb = getI32Value(0);
    ub = list.front();
    step = getI32Value(1);
  }

  auto symTable =
      builder.create<SymbolTable>("for-loop-line-" + std::to_string(loc.lineNo),
                                  parser.getCurrentSymbolTable());
  // Insert IV.

  auto ivSymbol =
      builder.create<Symbol>(iv, builder.getType<I32Type>(), symTable, true);

  symTable->insertSymbol(ivSymbol);

  auto body = parser.parseBlock(symTable);
  if (!body) {
    return nullptr;
  }

  return builder.create<ForStmt>(loc, ivSymbol, lb, ub, step, symTable, body);
}

IfStmt *IfStmt::parse(::mll::Parser &parser, ::mll::ASTBuilder &builder) {

  bool IsIfBlock = true;
  ExprList condList;
  llvm::SmallVector<SymbolTable *> symTableList;
  BlockList blockList;
  auto loc = parser.getCurrLoc();

  while (parser.isIdentifier("else") || IsIfBlock) {
    if (!IsIfBlock)
      parser.consumeToken();
    auto hasIf = parser.parseKeyword("if");
    loc = parser.getCurrLoc();
    if (hasIf || IsIfBlock) {
      assert(parser.isToken(Token::l_paren));
      auto condition = parser.parseExpr();
      if (!condition) {
        return nullptr;
      }
      if (condition->getExprType() != builder.getType<I1Type>()) {
        emitError(parser.getCurrLoc(),
                  "Expecting condition expression to be of i1 / bool type");
        return nullptr;
      }
      condList.push_back(condition);
    }
    auto symTable = builder.create<SymbolTable>("if-else-line-" +
                                                    std::to_string(loc.lineNo),
                                                parser.getCurrentSymbolTable());
    auto body = parser.parseBlock(symTable);
    if (!body) {
      return nullptr;
    }
    symTableList.push_back(symTable);
    blockList.push_back(body);
    IsIfBlock = false;
  }

  auto conditions =
      builder.create<ListExpr>(loc, builder.getType<ListType>(), condList);
  return builder.create<IfStmt>(loc, conditions, blockList, symTableList);
}

Expr *parseNoneTypeConstExpr(::mll::Parser &parser, ::mll::ASTBuilder &builder,
                             ::mll::Type *ty) {
  assert(false);
  return nullptr;
}

FuncStmt *FuncStmt::parse(::mll::Parser &parser, ::mll::ASTBuilder &builder) {

  auto loc = parser.getCurrLoc();

  auto funcNameOpt = parser.parseIdentifier();
  if (!funcNameOpt.has_value()) {
    emitError(parser.getCurrLoc(), "Expecting function name here");
    return nullptr;
  }

  auto funcName = funcNameOpt.value();

  if (!parser.parseToken(Token::l_paren)) {
    emitError(parser.getCurrLoc(), "Expecting ( while parsing function");
    return nullptr;
  }

  llvm::SmallVector<std::string> argNames;
  llvm::SmallVector<Type *> typeList;

  while (!parser.isToken(Token::r_paren)) {
    auto argNameOpt = parser.parseIdentifier();
    if (!argNameOpt.has_value()) {
      emitError(parser.getCurrLoc(), "Expecting function argument");
      return nullptr;
    }

    if (!parser.parseToken(Token::colon)) {
      emitError(parser.getCurrLoc(), "Expecting : after argument name");
      return nullptr;
    }

    argNames.push_back(argNameOpt.value());

    auto type = parser.parseType();
    if (!type) {
      return nullptr;
    }
    typeList.push_back(type);

    if (!parser.isToken(Token::r_paren)) {
      assert(parser.parseToken(Token::comma));
    }
  }

  assert(parser.parseToken(Token::r_paren));

  Type *returnType = builder.getType<NoneType>();
  if (parser.parseToken(Token::arrow)) {
    returnType = parser.parseType();
    if (!returnType) {
      return nullptr;
    }
  }

  auto funcType = builder.getType<FunctionType>(returnType, typeList);
  auto funcSymbol = builder.create<Symbol>(
      funcName, funcType, parser.getCurrentSymbolTable(), true);
  parser.insertSymbol(funcSymbol);

  auto symTable = builder.create<SymbolTable>(
      "func-" + funcName, parser.getCurrentSymbolTable(), true);

  llvm::SmallVector<Symbol *> args;
  for (unsigned I = 0; I < argNames.size(); ++I) {
    auto sym = builder.create<Symbol>(argNames[I], typeList[I], symTable, true);
    symTable->insertSymbol(sym);
    args.push_back(sym);
  }

  auto body = parser.parseBlock(symTable);
  if (!body) {
    return nullptr;
  }

  return builder.create<FuncStmt>(loc, funcName, args, symTable, funcType,
                                  body);
}

ReturnStmt *ReturnStmt::parse(::mll::Parser &parser,
                              ::mll::ASTBuilder &builder) {

  auto loc = parser.getCurrLoc();
  auto val = parser.parseToken(Token::l_paren);
  if (!val) {

    auto noneTy = builder.getType<NoneType>();
    return builder.create<ReturnStmt>(
        parser.getCurrLoc(),
        builder.create<NoneConstantExpr>(parser.getCurrLoc(), noneTy, 0));
  }

  auto expr = parser.parseExpr();
  if (!expr) {
    return nullptr;
  }
  assert(parser.parseToken(Token::r_paren));

  return builder.create<ReturnStmt>(loc, expr);
}

} // namespace builtin
} // namespace mll