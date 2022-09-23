//===- Parser.h - MLL Parser Interface --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MLIR Praser utils.
//
//===----------------------------------------------------------------------===//

#ifndef MLL_INCLUDE_PARSER_H
#define MLL_INCLUDE_PARSER_H
#include "mll/AST/AST.h"
#include "mll/Parse/ParserState.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>
#include <stack>

namespace mll {
class ASTModule;
class MLLContext;

template <class T> class ParserResult { T *value; };

inline mlir::InFlightDiagnostic emitError(Location loc,
                                          const llvm::Twine &msg) {
  mlir::Location mlirLoc = mlir::FileLineColLoc::get(
      loc.context->getMLIRContext(), loc.filename, loc.lineNo, loc.colNo);
  return mlir::emitError(mlirLoc, msg);
}

class Parser {

  struct OperatorInfo {
  private:
    union Value {
      BinaryOperator *op;
      Token::Kind kind;
    };
    Value value;
    bool isToken;

  public:
    explicit OperatorInfo(BinaryOperator *op) {
      value.op = op;
      isToken = false;
    }

    explicit OperatorInfo(Token::Kind kind) {
      value.kind = kind;
      isToken = true;
    }

    int getOpPrecedence() {
      assert(!isToken);
      return value.op->precedence;
    }

    std::string getOperatorKeyword() {
      assert(isOperator());
      return value.op->op;
    }

    bool isRelational() {
      assert(isOperator());
      return value.op->isRelational;
    }

    bool isOperator() { return !isToken; }

    bool isTokenKind() { return isToken; }

    bool isTokenKind(Token::Kind kind) { return isToken && value.kind == kind; }
  };

public:
  explicit Parser(ParserState &state, llvm::SourceMgr &srcMgr)
      : state(state), builder(state.context),
        diag(srcMgr, state.context->getMLIRContext()) {}

  ASTModule *parseASTModule();

  Block *parseBlock(SymbolTable *symTable);

  Stmt *parseStmt();

  Expr *parseExpr();

  bool isExprOperand(Token tok);

  Expr *parseExprOperand(Token::Kind prevToken);

  void pushOperation(std::stack<Expr *> &valueStack,
                     std::stack<OperatorInfo> &opsStack, Location loc);

  Expr *checkAndBuildExpr(Expr *lhs, Expr *rhs, OperatorInfo, Type *type);

  int getOpPrecedence(Parser::OperatorInfo op);

  bool isOperator(std::string id);

  Expr *parsePropertyExpr(Dialect::PropertyExprInfo &info);

  Expr *parseMethodExpr(Dialect::MethodExprInfo &info);

  Stmt *parseCustomStmt(Dialect::StmtInfo &info);

  Type *parseCustomType(Dialect::TypeInfo &info);

  Expr *parseConstantExpr(Type *ty);

  Expr *parseVariableExpr();

  ASTNode *parseDialectIdentifier(Dialect *dialect, StringRef id);

private:
  ParserState &state;
  ASTBuilder builder;

  mlir::SourceMgrDiagnosticHandler diag;

  /// Return the current token the parser is inspecting.
  const Token &getToken() const { return state.curToken; }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is.  This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  void consumeToken(Token::Kind kind) {
    assert(state.curToken.is(kind) && "consumed an unexpected token");
    consumeToken();
  }

  /// Get the location of the current token.
  Location getLoc() {
    return state.lex.getEncodedSourceLocation(getToken().getLoc());
  }

  Dialect *getDialect(StringRef str) { return state.context->getDialect(str); }

  Dialect *getBuiltinDialect() { return getDialect("builtin"); }

  bool isDialectName(StringRef str) { return getDialect(str) != nullptr; }

  bool isBuiltinDialectId(StringRef id) {
    auto builtin = getDialect("builtin");
    return (builtin->getIdentifierKind(id) != Dialect::None);
  }

  bool isFunctionName(StringRef str);

public:
  Symbol *getSymbol(StringRef variable) {
    for (auto &symTable : llvm::reverse(state.symbols)) {
      if (symTable->variables().find(variable.str()) !=
          symTable->variables().end()) {
        return symTable->variables()[variable.str()];
      }
      if (symTable->isIsolatedFromAboveSymTable()) {
        break;
      }
    }
    return nullptr;
  }

  void insertSymbol(Symbol *symbol) {
    assert(!state.symbols.empty());
    state.symbols.back()->insertSymbol(symbol);
  }

  StringRef getTokenSpelling() const { return state.curToken.getSpelling(); }

  bool isToken(Token::Kind k) const { return getToken().is(k); }

  bool isTokenSpelling(StringRef str) const {
    return (getTokenSpelling() == str);
  }

  /// Advance the current lexer onto the next token.
  void consumeToken() {
    assert(state.curToken.isNot(Token::eof, Token::error) &&
           "shouldn't advance past EOF or errors");
    state.curToken = state.lex.lexToken();
  }

  bool parseToken(Token::Kind kind) {
    if (isToken(kind)) {
      consumeToken();
      return true;
    }
    return false;
  }

  Type *parseType();

  llvm::Optional<int64_t> parseInteger() {
    if (!isToken(Token::integer)) {
      return llvm::None;
    }
    auto val = getToken().getIntegerValue();
    consumeToken();
    return val;
  }

  llvm::Optional<std::string> parseIdentifier() {
    if (!isToken(Token::identifier)) {
      return llvm::None;
    }
    std::string val = getTokenSpelling().str();
    consumeToken();
    return val;
  }

  bool isIdentifier(StringRef str) {
    if (!isToken(Token::identifier)) {
      return false;
    }
    return getTokenSpelling() == str;
  }

  llvm::Optional<double> parseFloat() {
    if (!isToken(Token::floatliteral)) {
      return llvm::None;
    }
    auto val = getToken().getFloatingPointValue();
    consumeToken();
    return val;
  }

  Location getCurrLoc() { return getLoc(); }

  // Search for any keyword from dialect context
  bool parseKeyword(StringRef kw) {
    if (!getToken().is(Token::identifier)) {
      return false;
    }

    if (getTokenSpelling() != kw) {
      return false;
    }
    consumeToken();
    return true;
  }

  SymbolTable *getCurrentSymbolTable() { return state.symbols.back(); }

  void pushSymbolTable(SymbolTable *sym) {
    assert(sym);
    state.symbols.push_back(sym);
  }

  SymbolTable *popSymbolTable() { return state.symbols.pop_back_val(); }
};

ASTModule *parseInputFile(std::unique_ptr<llvm::MemoryBuffer> file,
                          MLLContext *context);
} // namespace mll

#endif