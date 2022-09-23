//===- GPUDialectConversions.cpp - ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "mll/Dialect/GPU/GPUDialect.h"

#include "mll/AST/AST.h"
#include "mll/MLIRCodeGen/MLIRCodeGen.h"
#include "mll/Parse/BuiltinDialect.h"
#include "mll/Parse/Parser.h"
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>
using namespace mlir;

namespace mll {
namespace gpu {

class HostRegStmtConv : public StmtConversion {
  void convertStmt(mlir::OpBuilder &builder, mll::Stmt *stmt) override {
    auto hostStmt = llvm::cast<HostRegisterStmt>(stmt);
    auto args = hostStmt->args()->children();

    for (auto &arg : args) {
      auto argVal = convertExpr(arg).getLHSValue();
      auto type = argVal.getType().cast<MemRefType>();
      auto unrankedType =
          UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
      auto castVal =
          builder.create<memref::CastOp>(argVal.getLoc(), unrankedType, argVal);
      builder.create<mlir::gpu::HostRegisterOp>(toMLIRLoc(arg), castVal);
    }
  }
};

class LaunchStmtConv : public StmtConversion {
  void convertStmt(mlir::OpBuilder &builder, mll::Stmt *stmt) override {

    auto loc = toMLIRLoc(stmt);
    auto toIndex = [&](mlir::Value val) {
      return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                                val);
    };

    auto toI32 = [&](mlir::Value val) {
      return builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), val);
    };

    auto launchStmt = llvm::cast<LaunchStmt>(stmt);
    auto gridSize = launchStmt->blocks()->children();
    auto blockSize = launchStmt->threads()->children();

    Value gridX = toIndex(convertExpr(gridSize[0]).getValue());
    Value gridY = toIndex(convertExpr(gridSize[1]).getValue());
    Value gridZ = toIndex(convertExpr(gridSize[2]).getValue());
    Value blockX = toIndex(convertExpr(blockSize[0]).getValue());
    Value blockY = toIndex(convertExpr(blockSize[1]).getValue());
    Value blockZ = toIndex(convertExpr(blockSize[2]).getValue());

    auto launchOp = builder.create<mlir::gpu::LaunchOp>(
        loc, gridX, gridY, gridZ, blockX, blockY, blockZ);

    auto body = &launchOp.body().front();

    MLIRCodeGen::Scope scope(launchStmt->body(), launchStmt->symTable(), body);
    cg->pushScope(scope);

    builder.setInsertionPointToEnd(body);
    cg->setAllocaFor("blockIdx", toI32(launchOp.getBlockIds().x));
    cg->setAllocaFor("blockIdy", toI32(launchOp.getBlockIds().y));
    cg->setAllocaFor("blockIdz", toI32(launchOp.getBlockIds().z));
    cg->setAllocaFor("threadIdx", toI32(launchOp.getThreadIds().x));
    cg->setAllocaFor("threadIdy", toI32(launchOp.getThreadIds().y));
    cg->setAllocaFor("threadIdz", toI32(launchOp.getThreadIds().z));

    // Convert the block
    cg->convertBlock(false);

    // Move yield as last instruction.
    builder.setInsertionPointToEnd(body);
    builder.create<mlir::gpu::TerminatorOp>(loc, llvm::None);
    cg->popScope();
  }
};

void GPUDialect::registerMLIRConversions(::mll::MLIRCodeGen *cg) {
  cg->registerStmtConversion<HostRegisterStmt, HostRegStmtConv>();
  cg->registerStmtConversion<LaunchStmt, LaunchStmtConv>();
}
} // namespace gpu
} // namespace mll