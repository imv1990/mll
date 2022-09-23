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

#include "mll/MLIRCodeGen/MLIRCodeGen.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mll/AST/Dialect.h"
#include "mll/AST/MLLContext.h"

using namespace mll;

bool mll::convertToMLIR(mll::MLLContext *ctx,
                        mlir::OwningOpRef<mlir::ModuleOp> &module,
                        mll::ASTModule *astModule) {

  MLIRCodeGen codegen(module, ctx);
  return codegen.convertASTModule(astModule);
}

mlir::Value ReturnInfo::getLHSValue() {
  assert(kind == ValueKind);
  auto val = getValue();

  if (val.getType().isa<mlir::TensorType>()) {
    auto toTensorOp =
        llvm::cast<mlir::bufferization::ToTensorOp>(val.getDefiningOp());
    auto memref = toTensorOp.getMemref();
    toTensorOp->erase();
    return memref;
  }

  assert(false && "Unknown type for getLHSValue()");
}

void MLIRCodeGen::convertBlock(bool NeedReturnOp) {
  auto &scope = scopeStack.back();

  /// Get the parent function like operation where you can
  /// emit stack variables.
  // Always append the memrefs in the front of the block.
  auto stackBody = getFuncLikeOpBodyInStack();
  assert(stackBody);
  builder.setInsertionPointToStart(stackBody);

  // Register all the symbols as alloca.
  for (auto &sym : scope.symTable->variables()) {
    auto loc = toMLIRLoc(scope.block->getLoc());
    auto name = sym.first;
    auto type = sym.second->getType();
    // Do not allocate stack variable for loop IV.
    if (sym.second->isLoopIndVar()) {
      continue;
    }
    mlir::MemRefType memrefType;
    auto mlirType = convertType(type);
    if (mlirType.isa<mlir::MemRefType>()) {
      memrefType = mlirType.cast<mlir::MemRefType>();
    } else {
      memrefType = mlir::MemRefType::get(llvm::None, convertType(type));
    }

    auto memref = builder.create<mlir::memref::AllocaOp>(loc, memrefType);
    memref->setAttr("name", builder.getStringAttr(name));
    scope.mlirVarMap[name] = memref.getResult();
  }

  // Now start converting all the block statements.
  for (auto &stmt : scope.block->getStmts()) {
    builder.setInsertionPointToEnd(scope.mlirBlock);
    convertStmt(stmt);
  }

  if (NeedReturnOp)
    builder.create<mlir::func::ReturnOp>(toMLIRLoc(scope.block->getLoc()));
}

bool mll::MLIRCodeGen::convertASTModule(ASTModule *astModule) {

  auto loc = toMLIRLoc(astModule->getLoc());
  // create empty module.
  moduleOp = mlir::ModuleOp::create(loc);

  // create main function.
  auto funcType = builder.getFunctionType(llvm::None, llvm::None);
  auto func = mlir::func::FuncOp::create(loc, "main", funcType);
  moduleOp->push_back(func);

  auto block = func.addEntryBlock();
  Scope scope(astModule->getBlock(), astModule->getSymTable(), block, block);

  scopeStack.push_back(scope);

  // convert the block in current scope.
  convertBlock();

  scopeStack.pop_back();
  return true;
}

ReturnInfo MLIRConversion::convertExpr(mll::Expr *expr) {
  return cg->convertExpr(expr);
}

mlir::Type MLIRConversion::convertType(mll::Type *ty) {
  return cg->convertType(ty);
}

mlir::Location MLIRConversion::toMLIRLoc(mll::Location loc) {
  return cg->toMLIRLoc(loc);
}

mlir::Location MLIRConversion::toMLIRLoc(mll::ASTNode *node) {
  return cg->toMLIRLoc(node);
}