//===- OMPDialectConversions.cpp - ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "mll/Dialect/OMP/OMPDialect.h"

#include "mll/AST/AST.h"
#include "mll/MLIRCodeGen/MLIRCodeGen.h"
#include "mll/Parse/BuiltinDialect.h"
#include "mll/Parse/Parser.h"
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
using namespace mlir;

namespace mll {
namespace omp {

class ParallelStmtConv : public StmtConversion {
  void convertStmt(mlir::OpBuilder &builder, mll::Stmt *stmt) override {

    auto loc = toMLIRLoc(stmt);
    auto parallelStmt = llvm::cast<ParallelStmt>(stmt);
    auto parOp = builder.create<mlir::omp::ParallelOp>(loc);
    auto body = builder.createBlock(&parOp.region());
    MLIRCodeGen::Scope scope(parallelStmt->body(), parallelStmt->symTable(),
                             body);
    cg->pushScope(scope);
    // Convert the block
    cg->convertBlock(false);
    // Move yield as last instruction.
    builder.setInsertionPointToEnd(body);
    builder.create<mlir::omp::TerminatorOp>(loc, llvm::None);
    cg->popScope();
  }
};

class CriticalStmtConv : public StmtConversion {
  void convertStmt(mlir::OpBuilder &builder, mll::Stmt *stmt) override {

    auto loc = toMLIRLoc(stmt);
    auto parallelStmt = llvm::cast<CriticalStmt>(stmt);

    auto symbolName = parallelStmt->symTable()->getName();
    {

      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(cg->getModule().getBody());
      builder.create<mlir::omp::CriticalDeclareOp>(
          loc, builder.getStringAttr(symbolName), builder.getI64IntegerAttr(0));
    }
    auto parOp = builder.create<mlir::omp::CriticalOp>(
        loc, mlir::FlatSymbolRefAttr::get(builder.getContext(), symbolName));
    auto body = builder.createBlock(&parOp.region());
    MLIRCodeGen::Scope scope(parallelStmt->body(), parallelStmt->symTable(),
                             body);
    cg->pushScope(scope);
    // Convert the block
    cg->convertBlock(false);
    // Move yield as last instruction.
    builder.setInsertionPointToEnd(body);
    builder.create<mlir::omp::TerminatorOp>(loc, llvm::None);
    cg->popScope();
  }
};

void OMPDialect::registerMLIRConversions(::mll::MLIRCodeGen *cg) {
  cg->registerStmtConversion<ParallelStmt, ParallelStmtConv>();
  cg->registerStmtConversion<CriticalStmt, CriticalStmtConv>();
}
} // namespace omp
} // namespace mll