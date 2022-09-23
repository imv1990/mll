//===- Expr.h - MLL dialect definitions generator -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DialectGen uses the description of dialects to generate C++ definitions.
//
//===----------------------------------------------------------------------===//

#include "Type.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <cstddef>

class Expr {
  llvm::Record *def;

public:
  Expr(llvm::Record *def) : def(def) {}

  ArrayRef<SMLoc> getLoc() { return def->getLoc(); }

  std::string getExprKind() {

    if (def->isSubClassOf("ConstantExpr")) {
      return "::mll::Expr::ConstantExprKind";
    }

    if (def->isSubClassOf("MethodExpr")) {
      return "::mll::Expr::MethodExprKind";
    }

    if (def->isSubClassOf("PropertyExpr")) {
      return "::mll::Expr::PropertyExprKind";
    }

    if (def->isSubClassOf("TypeExprKind")) {
      return "::mll::Expr::TypeExprKind";
    }

    if (def->isSubClassOf("VariableExpr")) {
      return "::mll::Expr::VariableExprKind";
    }

    return "::mll::Expr::GenericExprKind";
  }

  bool isSubClassOf(StringRef className) {
    return def->isSubClassOf(className);
  }

  std::string getUniqueName() {
    return getDialect().getName().str() + "." + getName().str();
  }

  StringRef getName() { return def->getValueAsString("name"); }

  StringRef getCppClassName() { return def->getValueAsString("cppClassName"); }

  Dialect getDialect() { return Dialect(def->getValueAsDef("dialect")); }

  ArgumentDAG getArguments() {

    return ArgumentDAG(def->getValueAsDag("arguments"), def->getLoc(), false);
  }

  ArgumentDAG getMethodArguments() {
    assert(isSubClassOf("MethodExpr"));
    return ArgumentDAG(def->getValueAsDag("methodArgs"), def->getLoc(), true);
  }

  StringRef getExtraClassDecl() {
    return def->getValueAsString("extraClassDeclaration");
  }

  Type getVariableType() {
    assert(isVariableExpr());
    return Type(def->getValueAsDef("variableType"));
  }

  Type getConstantType() {
    assert(isConstantExpr());
    return Type(def->getValueAsDef("constantType"));
  }

  bool needParserMethod() { return def->getValueAsBit("declareParserMethod"); }

  CPPType getReturnType() { return CPPType(def->getValueAsDef("exprType")); }

  std::string getFullClassName() {
    return ::getFullClassName(getDialect().getName(), getCppClassName());
  }

  std::string getTypeIDExpr() {
    return ::getTypeIDForClass(getFullClassName());
  }

  // Valid of builtin method/ property expressions only.
  StringRef getInferReturnTypeBody() {
    return def->getValueAsString("inferReturnTypeBody");
  }

  bool isBuiltinMethodOrProp() {
    return isSubClassOf("PropertyExpr") || isSubClassOf("MethodExpr");
  }

  bool isConstantExpr() { return isSubClassOf("ConstantExpr"); }

  bool isVariableExpr() { return isSubClassOf("VariableExpr"); }

  bool isBuiltinMethod() { return isSubClassOf("MethodExpr"); }

  bool isPropertyExpr() { return isSubClassOf("PropertyExpr"); }
};

class Stmt {
  llvm::Record *def;

public:
  Stmt(llvm::Record *def) : def(def) {}

  StringRef getCppClassName() { return def->getValueAsString("cppClassName"); }

  Dialect getDialect() { return Dialect(def->getValueAsDef("dialect")); }

  ArgumentDAG getArguments() {
    return ArgumentDAG(def->getValueAsDag("arguments"), def->getLoc(), false);
  }

  StringRef getExtraClassDecl() {
    return def->getValueAsString("extraClassDeclaration");
  }

  bool needParserMethod() { return def->getValueAsBit("declareParserMethod"); }

  std::string getFullClassName() {
    return ::getFullClassName(getDialect().getName(), getCppClassName());
  }

  StringRef getName() { return def->getValueAsString("name"); }

  std::string getUniqueName() {
    return getDialect().getName().str() + "." + getName().str();
  }

  std::string getTypeIDExpr() {
    return ::getTypeIDForClass(getFullClassName());
  }
};

class BinaryOperator {
  llvm::Record *def;

public:
  BinaryOperator(llvm::Record *def) : def(def) {}

  Dialect getDialect() { return Dialect(def->getValueAsDef("dialect")); }

  StringRef getKeyword() { return def->getValueAsString("kw"); }

  bool isRelational() { return def->getValueAsBit("isRelationalKind"); }

  int getPrecedence() { return def->getValueAsInt("precedence"); }
};