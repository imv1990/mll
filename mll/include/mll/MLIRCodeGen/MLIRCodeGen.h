//===- MLIRCodeGen.h - MLL Parser Interface ----------- ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MLL AST to MLIR codegen.
//
//===----------------------------------------------------------------------===//

#ifndef MLL_INCLUDE_MLIR_CODEGEN_H
#define MLL_INCLUDE_MLIR_CODEGEN_H

#include "mll/AST/AST.h"
#include "mll/AST/MLLContext.h"
#include "mll/AST/Type.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"

#include <string>

namespace mll {

class MLIRCodeGen;

class ReturnInfo {

  mlir::Value val = {};
  mlir::Operation *op = nullptr;
  mlir::Attribute attr = {};
  mlir::Type type = {};

  enum Kind {
    None = 0,
    ValueKind,
    OpKind,
    AttrKind,
    TypeKind,
  };

  Kind kind;

public:
  ReturnInfo() : kind(None) {}
  ReturnInfo(mlir::Value val) : val(val), kind(ValueKind) {}
  ReturnInfo(mlir::Operation *val) : op(val), kind(OpKind) {}

  mlir::Value getValue() {
    assert(kind == ValueKind);
    return val;
  }

  mlir::Value getLHSValue();

  mlir::Operation *getOperation() {
    assert(kind == OpKind);
    return op;
  }
};

class MLIRConversion {
protected:
  MLIRCodeGen *cg;
  friend class MLIRCodeGen;

  ReturnInfo convertExpr(mll::Expr *);

  mlir::Type convertType(mll::Type *);

  mlir::Location toMLIRLoc(mll::Location);

  mlir::Location toMLIRLoc(mll::ASTNode *);

  virtual ~MLIRConversion() {}
};

class TypeConversion : public MLIRConversion {
public:
  virtual mlir::Type convertType(mlir::OpBuilder &builder, mll::Type *ty) = 0;
};

class ExprConversion : public MLIRConversion {
public:
  virtual ReturnInfo convertExpr(mlir::OpBuilder &builder, mll::Expr *expr) = 0;
};

class StmtConversion : public MLIRConversion {
public:
  virtual void convertStmt(mlir::OpBuilder &builder, mll::Stmt *stmt) = 0;
};

class MLIRCodeGen {
public:
  struct Scope {
    Block *block;
    mlir::Block *mlirBlock;
    SymbolTable *symTable;
    llvm::StringMap<mlir::Value> mlirVarMap;

    // Mostly used for generating stack variables.
    mlir::Block *parentFuncLikeOpBody = nullptr;

    Scope(Block *block, SymbolTable *symTable, mlir::Block *mlirBlock,
          mlir::Block *funcLikeOpBody = nullptr)
        : block(block), mlirBlock(mlirBlock), symTable(symTable),
          parentFuncLikeOpBody(funcLikeOpBody) {}

    bool isFuncLikeOpBody() { return parentFuncLikeOpBody != nullptr; }
  };

  mlir::Location toMLIRLoc(mll::Location loc) {
    return mlir::FileLineColLoc::get(mlirCtx, loc.filename, loc.lineNo,
                                     loc.colNo);
  }

  mlir::Location toMLIRLoc(mll::ASTNode *stmt) {
    return toMLIRLoc(stmt->getLoc());
  }

  mlir::Type convertType(mll::Type *ty) {
    auto name = ty->getUniqueName();
    auto iter = typeConvRegistry.find(name);
    if (iter == typeConvRegistry.end()) {
      llvm::report_fatal_error(
          StringRef("No conversion registered for type " + name), false);
    }
    auto typeConvClass = iter->second;
    return typeConvClass->convertType(builder, ty);
  }

  ReturnInfo convertExpr(mll::Expr *expr) {
    auto name = expr->getUniqueName();
    auto iter = exprConvRegistry.find(name);
    if (iter == exprConvRegistry.end()) {
      llvm::report_fatal_error(
          StringRef("No conversion registered for expression " + name), false);
    }
    auto exprConvClass = iter->second;
    return exprConvClass->convertExpr(builder, expr);
  }

  void convertStmt(mll::Stmt *stmt) {
    auto name = stmt->getUniqueName();
    auto iter = stmtConvRegistry.find(name);
    if (iter == stmtConvRegistry.end()) {
      llvm::report_fatal_error(
          StringRef("No conversion registered for statement " + name), false);
    }
    auto stmtConvClass = iter->second;
    stmtConvClass->convertStmt(builder, stmt);
  }

  void convertBlock(bool NeedReturnOp = true);

  bool convertASTModule(ASTModule *module);

  mlir::Block *getFuncLikeOpBodyInStack() {
    for (auto &scope : llvm::reverse(scopeStack)) {
      if (scope.isFuncLikeOpBody()) {
        return scope.parentFuncLikeOpBody;
      }
    }
    assert(false && "Could not find function like op for stack variables");
    return nullptr;
  }

  mlir::Value getAllocaFor(StringRef name) {
    for (auto &scope : llvm::reverse(scopeStack)) {
      auto iter = scope.mlirVarMap.find(name);
      if (iter == scope.mlirVarMap.end()) {
        if (scope.symTable->isIsolatedFromAboveSymTable()) {
          break;
        }
        continue;
      }
      return iter->second;
    }
    assert(false && "could not find the variable declaration");
    return {};
  }

  MLIRCodeGen(mlir::OwningOpRef<mlir::ModuleOp> &module, MLLContext *ctx)
      : mllCtx(ctx), mlirCtx(mllCtx->getMLIRContext()), builder(mlirCtx),
        moduleOp(module) {
    registerConversions();
  }

  template <class TypeT, class TypeConvT> void registerTypeConversion() {
    auto instance = new (mllCtx, alignof(TypeConvT)) TypeConvT();
    instance->cg = this;
    typeConvRegistry[TypeT::getUniqueName()] = instance;
  }

  template <class TypeT, class TypeConvT> void registerStmtConversion() {
    auto instance = new (mllCtx, alignof(TypeConvT)) TypeConvT();
    instance->cg = this;
    stmtConvRegistry[TypeT::getUniqueName()] = instance;
  }

  template <class TypeT, class TypeConvT> void registerExprConversion() {
    auto instance = new (mllCtx, alignof(TypeConvT)) TypeConvT();
    instance->cg = this;
    exprConvRegistry[TypeT::getUniqueName()] = instance;
  }

  void registerConversions() {
    for (auto *dialect : mllCtx->getLoadedDialects()) {
      dialect->registerMLIRConversions(this);
    }
  }

  void setAllocaFor(StringRef name, mlir::Value val) {
    scopeStack.back().mlirVarMap[name] = val;
  }

  void pushScope(Scope scope) {
    scopeStack.push_back(scope);
    builder.setInsertionPointToEnd(scope.mlirBlock);
  }

  Scope popScope() {
    auto outScope = scopeStack.pop_back_val();
    builder.setInsertionPointToEnd(scopeStack.back().mlirBlock);
    return outScope;
  }

  mlir::ModuleOp getModule() { return moduleOp.get(); }

private:
  mll::MLLContext *mllCtx;
  mlir::MLIRContext *mlirCtx;
  mlir::OpBuilder builder;
  llvm::SmallVector<Scope> scopeStack;
  mlir::OwningOpRef<mlir::ModuleOp> &moduleOp;
  llvm::StringMap<TypeConversion *> typeConvRegistry;
  llvm::StringMap<ExprConversion *> exprConvRegistry;
  llvm::StringMap<StmtConversion *> stmtConvRegistry;
}; // namespace mll

bool convertToMLIR(mll::MLLContext *ctx,
                   mlir::OwningOpRef<mlir::ModuleOp> &module,
                   mll::ASTModule *astModule);
} // namespace mll
#endif