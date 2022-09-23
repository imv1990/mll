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

#ifndef MLL_INCLUDE_AST_MLLDIALECT_H
#define MLL_INCLUDE_AST_MLLDIALECT_H

#include "mlir/Support/LLVM.h"
#include "mll/AST/MLLContext.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <functional>
#include <mlir/Support/TypeID.h>
#include <string>

using llvm::ArrayRef;
using mlir::SelfOwningTypeID;
using mlir::StringRef;
using mlir::TypeID;

namespace mll {

class MLIRCodeGen;

class Dialect {
public:
  typedef void (*MethodPtr)(void);

  enum IdentifierKind {
    None,
    TypeKind,
    PropertyExprKind,
    MethodExprKind,
    ConstantExprKind,
    StmtKind,
  };

  struct StmtInfo {
    std::string name;
    MethodPtr parserFn;
    bool hasCustomParseMethod;
  };

  struct TypeInfo {
    std::string name;
    MethodPtr parserFn;
    bool hasCustomParseMethod;
  };

  struct PropertyExprInfo {
    std::string name;
    MethodPtr buildFn;
  };

  struct MethodExprInfo {
    std::string name;
    MethodPtr buildFn;
  };

  struct ConstantExprInfo {
    std::string name;
    MethodPtr parserFn;
  };

  struct VariableExprInfo {
    std::string name;
    MethodPtr parserFn;
  };

protected:
  mlir::TypeID id;
  mll::MLLContext *context;
  StringRef name;

  // Contains names of all Types, AST Nodes.
  llvm::StringMap<IdentifierKind> dialectSymbols;

  llvm::StringMap<TypeInfo> typeInfoMap;

  llvm::StringMap<PropertyExprInfo> propExprMap;

  llvm::StringMap<MethodExprInfo> methodExprMap;

  llvm::StringMap<ConstantExprInfo> constExprMap;

  llvm::StringMap<VariableExprInfo> varExprMap;

  llvm::StringMap<StmtInfo> stmtMap;

  void throwErrorIfExists(std::string symName, IdentifierKind kind) {
    if (dialectSymbols.find(symName) != dialectSymbols.end()) {
      llvm::report_fatal_error("Registering for duplicte dialect symbol " +
                               symName + " in dialect " + name);
    }

    dialectSymbols[symName] = kind;
  }

protected:
  Dialect(StringRef name, mll::MLLContext *context, mlir::TypeID id)
      : id(id), context(context), name(name) {}

public:
  mlir::TypeID getTypeID() const { return id; }

  StringRef getName() { return name; }

  IdentifierKind getIdentifierKind(StringRef id) {
    auto iter = dialectSymbols.find(id);
    if (iter == dialectSymbols.end()) {
      return Dialect::None;
    }
    return iter->second;
  }

  PropertyExprInfo getPropertyExprInfo(StringRef name) {
    assert(propExprMap.find(name) != propExprMap.end());
    return propExprMap[name];
  }

  MethodExprInfo getMethodExprInfo(StringRef name) {
    assert(methodExprMap.find(name) != methodExprMap.end());
    return methodExprMap[name];
  }

  StmtInfo getStmtInfo(StringRef name) {
    assert(stmtMap.find(name) != stmtMap.end());
    return stmtMap[name];
  }

  ConstantExprInfo getConstantExprInfo(StringRef name) {
    assert(constExprMap.find(name) != constExprMap.end());
    return constExprMap[name];
  }

  bool hasVariableExprInfo(StringRef name) {
    return varExprMap.find(name) != varExprMap.end();
  }

  VariableExprInfo getVariableExprInfo(StringRef name) {
    assert(varExprMap.find(name) != varExprMap.end());
    return varExprMap[name];
  }

  TypeInfo getTypeInfo(StringRef name) {
    assert(typeInfoMap.find(name) != typeInfoMap.end());
    return typeInfoMap[name];
  }

  virtual void registerMLIRConversions(MLIRCodeGen *cg) = 0;

protected:
  template <class TypeT> void registerType() {

    throwErrorIfExists(TypeT::getName(), TypeKind);
    TypeInfo t;
    t.name = TypeT::getName();
    t.hasCustomParseMethod = TypeT::hasCustomParseMethod();
    if (t.hasCustomParseMethod) {
      t.parserFn = (MethodPtr)TypeT::parse;
    } else {
      t.parserFn = nullptr;
    }

    typeInfoMap[t.name] = t;
  }

  template <class StmtT> void registerStmt() {

    throwErrorIfExists(StmtT::getName(), StmtKind);
    StmtInfo t;
    t.name = StmtT::getName();
    t.hasCustomParseMethod = StmtT::hasCustomParseMethod();
    if (t.hasCustomParseMethod) {
      t.parserFn = (MethodPtr)StmtT::parse;
    } else {
      t.parserFn = nullptr;
    }

    stmtMap[t.name] = t;
  }

  template <class ExprT> void registerPropertyExpr() {
    throwErrorIfExists(ExprT::getName(), PropertyExprKind);
    PropertyExprInfo info;
    info.name = ExprT::getName();
    info.buildFn = (MethodPtr)ExprT::build;

    propExprMap[info.name] = info;
  }

  template <class ExprT> void registerMethodExpr() {
    throwErrorIfExists(ExprT::getName(), MethodExprKind);
    MethodExprInfo info;
    info.name = ExprT::getName();
    info.buildFn = (MethodPtr)ExprT::build;

    methodExprMap[info.name] = info;
  }

  template <class ExprT> void registerConstantExprTypeParsing(MethodPtr ptr) {
    ConstantExprInfo info;
    info.name = ExprT::getName();
    info.parserFn = ptr;

    constExprMap[info.name] = info;
  }

  template <class ExprT> void registerVariableExprTypeParsing(MethodPtr ptr) {
    VariableExprInfo info;
    info.name = ExprT::getName();
    info.parserFn = ptr;

    varExprMap[info.name] = info;
  }

  virtual void registerTypes() = 0;

  virtual void registerBinaryOperators() {}

  virtual void registerPropertyExprs() {}

  virtual void registerMethodExprs() {}

  virtual void registerStmts() {}

  virtual void registerConstantExprTypes() {}

  virtual void registerVariableExprTypes() {}

  // virtual void registerStmt() = 0;

  // virtual void registerExpr() = 0;

  virtual ~Dialect() {}
};

} // namespace mll

#endif