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

#ifndef MLL_INCLUDE_AST_AST_H
#define MLL_INCLUDE_AST_AST_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/TypeID.h"

#include "mll/AST/Dialect.h"
#include "mll/AST/MLLContext.h"
#include "mll/AST/Type.h"

#include <string>
#include <utility>
#include <vector>

namespace mll {

class ASTBuilder;
using mlir::TypeID;

struct Location {
  llvm::StringRef filename;
  unsigned lineNo, colNo;
  MLLContext *context;

  static Location get(MLLContext *c, llvm::StringRef file, unsigned lineNo,
                      unsigned colNo) {
    Location loc;
    loc.filename = file;
    loc.lineNo = lineNo;
    loc.colNo = colNo;
    loc.context = c;
    return loc;
  }
};

class Stmt;
typedef llvm::SmallVector<Stmt *, 2> StmtList;
class Expr;
typedef llvm::SmallVector<Expr *, 2> ExprList;

class Block {
  StmtList nodes;
  Location loc;

  friend ASTBuilder;
  explicit Block(Location loc, StmtList &nodes) : nodes(nodes), loc(loc) {}

public:
  mll::Location getLoc() { return loc; }

  StmtList &getStmts() { return nodes; }

  void dump(llvm::raw_ostream &os) const;
};

typedef llvm::SmallVector<Block *> BlockList;

class Expr;
namespace builtin {
class ListExpr;
}
typedef llvm::ArrayRef<Expr *> ExprListRef;
typedef llvm::ArrayRef<Block *> BlockListRef;

class ASTNode {
public:
  enum Kind {
    StmtKind = 1,
    ExprKind = 2,
  };

private:
  Location loc;
  Kind k;
  TypeID id;
  DialectNamePair pair;
  ExprList nodeList;
  BlockList blockList;

protected:
  explicit ASTNode(Location loc, Kind k, TypeID id, DialectNamePair pair,
                   ExprListRef nodes, BlockListRef blocks)
      : loc(loc), k(k), id(id), pair(pair),
        nodeList(nodes.begin(), nodes.end()),
        blockList(blocks.begin(), blocks.end()) {}

public:
  ExprList children() const { return nodeList; }

  ExprList &children() { return nodeList; }

  Expr *getExpr(unsigned I) const { return nodeList[I]; }

  Block *getBlock(unsigned I) const { return blockList[I]; }

  BlockList &blocks() { return blockList; }

  BlockList blocks() const { return blockList; }

  Location getLoc() const { return loc; }

  Kind getKind() const { return k; }

  TypeID getTypeID() const { return id; }

  DialectNamePair getDialectNamePair() { return pair; }

  Dialect *getDialect(MLLContext *ctx) const {
    auto d = ctx->getDialect(pair.first);
    assert(d);
    return d;
  }

  StringRef getName() const { return pair.second; }

  std::string getUniqueName() const { return pair.first + "." + pair.second; }

  virtual void dump(llvm::raw_ostream &os) const = 0;

  virtual ~ASTNode() {}
};

class Stmt : public ASTNode {
protected:
  explicit Stmt(Location loc, TypeID id, DialectNamePair pair,
                ExprListRef nodes, BlockListRef blocks)
      : ASTNode(loc, ASTNode::StmtKind, id, pair, nodes, blocks) {}

public:
  static bool classof(const ASTNode *base) {
    return base->getKind() == ASTNode::StmtKind;
  }
};

class Expr : public ASTNode {
public:
  Type *getExprType() const { return ty; }

  static bool classof(const ASTNode *base) {
    return base->getKind() == ASTNode::ExprKind;
  }

  enum ExprKind {
    GenericExprKind = 0,
    ConstantExprKind,
    MethodExprKind,
    PropertyExprKind,
    TypeExprKind,
    VariableExprKind,
  };

  enum ExprKind getKind() { return kind; }

  bool isConstantExpr() { return kind == ConstantExprKind; }

protected:
  explicit Expr(Location loc, TypeID id, DialectNamePair pair, Type *ty,
                ExprListRef nodes, BlockListRef blocks, enum ExprKind kind)
      : ASTNode(loc, ASTNode::ExprKind, id, pair, nodes, blocks), ty(ty),
        kind(kind) {}

private:
  Type *ty;
  enum ExprKind kind;
};

class SymbolTable;

class Symbol {
private:
  std::string name;
  Type *type;
  SymbolTable *parent;
  bool isLoopIV;

  friend class SymbolTable;
  friend class ASTBuilder;
  explicit Symbol(StringRef name, Type *type, SymbolTable *parent,
                  bool isLoopIV = false)
      : name(name), type(type), parent(parent), isLoopIV(isLoopIV) {}

public:
  void dump(llvm::raw_ostream &os) const;

  StringRef getName() { return name; }

  void setType(Type *typeVal) { type = typeVal; }

  Type *getType() const { return type; }

  bool isLoopIndVar() const { return isLoopIV; }
};

class SymbolTable {
private:
  std::string name;

  std::map<std::string, Symbol *> variableMap;

  llvm::StringMap<Type *> typeAliases;

  llvm::StringSet<> dialects;

  SymbolTable *parent;

  llvm::SmallVector<SymbolTable *, 2> children;

  bool isIsolatedFromAbove;

  friend class ASTBuilder;
  explicit SymbolTable(StringRef name, SymbolTable *parent,
                       bool isIsolatedFromAbove = false)
      : name(name), parent(parent), isIsolatedFromAbove(isIsolatedFromAbove) {}

public:
  StringRef getName() const { return name; }

  bool isIsolatedFromAboveSymTable() const { return isIsolatedFromAbove; }

  std::map<std::string, Symbol *> &variables() { return variableMap; }

  void insertSymbol(Symbol *symbol) {
    assert(variableMap.find(symbol->getName().str()) == variableMap.end());
    variableMap[symbol->getName().str()] = symbol;
  }

  SymbolTable *getParent() { return parent; }

  void dump(llvm::raw_ostream &os) const;
};

struct TypeExprInfo {
  union Value {
    ::mll::Expr *expr;
    ::mll::Type *type;
  };
  Value value;
  bool isType;
};

// Class which holds the AST structure of input file.
class ASTModule {
private:
  Block *body;
  Location loc;
  SymbolTable *symTable;

  friend ASTBuilder;
  explicit ASTModule(Location loc, Block *body, SymbolTable *table)
      : body(body), loc(loc), symTable(table) {}

public:
  void dump(llvm::raw_ostream &OS) {
    llvm::formatted_raw_ostream fos(OS);
    body->dump(fos);
  }

  mll::Location getLoc() { return loc; }

  Block *getBlock() { return body; }

  SymbolTable *getSymTable() { return symTable; }
};

/// Builder class for all AST nodes.
class ASTBuilder {
private:
  MLLContext *context;

public:
  explicit ASTBuilder(MLLContext *context) : context(context) {}

  /// Get the hash code for the TypeID and args passed to build the type.
  // `code` uniquely identifies every type and duplicates type have same hash.
  template <class TypeT, typename... Params> TypeT *getType(Params &&... args) {
    llvm::hash_code code =
        llvm::hash_combine(TypeID::get<TypeT>(), TypeT::getUniqueName(),
                           std::forward<Params>(args)...);

    auto tyItr = context->typeStorageMap.find(code);
    if (tyItr != context->typeStorageMap.end()) {
      return llvm::dyn_cast<TypeT>(tyItr->second);
    }

    TypeT *ty = new (context, alignof(TypeT))
        TypeT(context, std::forward<Params>(args)...);
    context->typeStorageMap[code] = ty;
    return ty;
  }

  template <class Node, typename... Params> Node *create(Params &&... args) {
    return new (context, alignof(Node)) Node(std::forward<Params>(args)...);
  }
};
} // namespace mll

#define OVERLOAD_OSTREAM_OPERATOR(T)                                           \
  inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const T &t) {    \
    t.dump(os);                                                                \
    return os;                                                                 \
  }

#define OVERLOAD_OSTREAM_VEC_OPERATOR(T)                                       \
  inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,                  \
                                       const llvm::SmallVector<T *> &val) {    \
    os << "[";                                                                 \
    for (unsigned I = 0; I < val.size(); ++I) {                                \
      if (I != 0) {                                                            \
        os << ",";                                                             \
      }                                                                        \
      os << *val[I];                                                           \
    }                                                                          \
    os << "]";                                                                 \
    return os;                                                                 \
  }

OVERLOAD_OSTREAM_OPERATOR(mll::Symbol)
OVERLOAD_OSTREAM_OPERATOR(mll::SymbolTable)
OVERLOAD_OSTREAM_OPERATOR(mll::Type)
OVERLOAD_OSTREAM_OPERATOR(mll::Block)
OVERLOAD_OSTREAM_OPERATOR(mll::Expr)
OVERLOAD_OSTREAM_OPERATOR(mll::Stmt)

OVERLOAD_OSTREAM_VEC_OPERATOR(mll::SymbolTable)
OVERLOAD_OSTREAM_VEC_OPERATOR(mll::Block)
OVERLOAD_OSTREAM_VEC_OPERATOR(mll::Symbol)
OVERLOAD_OSTREAM_VEC_OPERATOR(mll::Type)

#endif