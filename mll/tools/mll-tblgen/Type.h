//===- Type.h - MLL dialect definitions generator -------------------------===//
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

#include "Dialect.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"

inline std::string getTypeIDForClass(StringRef className) {
  return "::mlir::TypeID::get<" + className.str() + ">()";
}

inline std::string getFullClassName(StringRef dialectName,
                                    StringRef className) {
  return "::mll::" + dialectName.str() + "::" + className.str();
}

class CPPType {
  llvm::Record *def;

public:
  CPPType(llvm::Record *def) : def(def) {}

  // FIXME: Add space because of a bug in Class.h
  std::string getTypeStr() { return def->getValueAsString("c").str() + " "; }

  std::string getTypeStrWithoutPtr() {
    auto c = def->getValueAsString("c");
    if (c.back() == '*') {
      auto newC = c.drop_back();
      return newC.str() + " ";
    }
    return getTypeStr();
  }

  bool isVectorType() { return def->isSubClassOf("VectorCPPType"); }

  bool isMLLCPPType() { return def->isSubClassOf("MLLCPPType"); }

  bool isMLLTypeConstraint() { return def->isSubClassOf("TypeConstraint"); }

  bool isTypeArgument() { return def->isSubClassOf("TypeArg"); }

  bool isMLLExprClass() { return getTypeStr() == "::mll::Expr* "; }

  bool isMLLExpr() { return def->isSubClassOf("MLLExprType"); }

  bool isBlock() { return getTypeStr() == "::mll::Block* "; }

  bool isBlockList() {
    if (!isVectorType()) {
      return false;
    }
    CPPType type = CPPType(def->getValueAsDef("baseTy"));

    return type.isBlock();
  }

  bool isSymbol() { return getTypeStr() == "::mll::Symbol* "; }

  bool isSymbolTable() { return getTypeStr() == "::mll::SymbolTable* "; }

  bool canStoreInASTNode() { return isBlock() || isMLLExpr(); }

  bool isASTNodeRelated() {
    return isMLLExpr() || isMLLTypeConstraint() || isBlock() || isSymbol() ||
           isTypeArgument() || isMLLCPPType() || isSymbolTable();
  }
};

struct NodeInfo {
  CPPType type;
  unsigned argNum; // ASTNode argNum
  StringRef name;
  unsigned index; // argument number in dag.

  explicit NodeInfo(CPPType _type, unsigned num, StringRef n, unsigned indexVal)
      : type(_type), argNum(num), name(n), index(indexVal) {}
};

class ArgumentDAG {
  DagInit *dag;
  ArrayRef<SMLoc> locs;
  bool isMehtodExprArgs;

public:
  BitVector astNodes;
  SmallVector<NodeInfo, 2> expressions;
  SmallVector<NodeInfo, 2> blocks;
  SmallVector<NodeInfo, 2> others;
  SmallVector<NodeInfo, 2> typeArguments; // Valid only for method expressions

  explicit ArgumentDAG(DagInit *dag, ArrayRef<SMLoc> locs,
                       bool isMehtodExprArgs)
      : dag(dag), locs(locs), isMehtodExprArgs(isMehtodExprArgs) {

    unsigned blockNum = 0;
    unsigned exprNum = 0;
    unsigned typeArgNum = 0;
    for (unsigned I = 0; I < dag->getNumArgs(); ++I) {
      if (isTypeConstraint(I)) {
        others.push_back(NodeInfo(CPPType(nullptr), 0, getArgName(I), I));
        astNodes.push_back(false);
        continue;
      }
      auto argType = getArgType(I);
      if (argType.isMLLExpr()) {
        expressions.push_back(NodeInfo(argType, blockNum++, getArgName(I), I));
        astNodes.push_back(true);
        continue;
      }
      if (argType.isBlock() || argType.isBlockList()) {
        astNodes.push_back(true);
        blocks.push_back(NodeInfo(argType, exprNum++, getArgName(I), I));
        continue;
      }
      if (argType.isTypeArgument()) {
        if (!isMehtodExprArgs) {
          llvm::PrintFatalError(
              "Type as arguments are allowed in MethodExpr only");
        }
        astNodes.push_back(false);
        typeArguments.push_back(
            NodeInfo(argType, typeArgNum++, getArgName(I), I));
        continue;
      }
      others.push_back(NodeInfo(argType, 0, getArgName(I), I));
      astNodes.push_back(false);
    }
  }

  std::string getArgFieldName(unsigned i) {
    if (astNodes[i]) {
      return getArgName(i).str() + "()";
    }
    return "_" + getArgName(i).str();
  }

  unsigned getNumArgs() { return dag->getNumArgs(); }

  CPPType getArgType(unsigned i) {
    auto arg = dag->getArg(i);
    auto defInit = dyn_cast<llvm::DefInit>(arg);
    if (!defInit || !defInit->getDef()->isSubClassOf("CPPType")) {
      assert(false);
      llvm::PrintFatalError(
          locs, "Invalid type argument. Expected object of CPPType");
    }
    return CPPType(defInit->getDef());
  }

  bool isTypeConstraint(unsigned I) {
    auto arg = dag->getArg(I);
    auto defInit = dyn_cast<llvm::DefInit>(arg);
    return (defInit && defInit->getDef()->isSubClassOf("TypeConstraint"));
  }

  StringRef getArgName(unsigned i) { return dag->getArgName(i)->getValue(); }
};

class Type {
  llvm::Record *def;

public:
  Type(llvm::Record *def) : def(def) {}

  bool needParserMethod() { return def->getValueAsBit("declareParserMethod"); }

  ArgumentDAG getArguments() {
    return ArgumentDAG(def->getValueAsDag("parameters"), def->getLoc(), false);
  }

  std::string getTypeIDExpr() {
    return ::getTypeIDForClass(getFullClassName());
  }

  StringRef getCppClassName() { return def->getValueAsString("cppClassName"); }

  Dialect getDialect() { return Dialect(def->getValueAsDef("dialect")); }

  DagInit *getParameters() { return def->getValueAsDag("parameters"); }

  StringRef getName() { return def->getValueAsString("name"); }

  std::string getUniqueName() {
    return getDialect().getName().str() + "." + getName().str();
  }

  std::string getFullClassName() {
    return ::getFullClassName(getDialect().getName(), getCppClassName());
  }
};