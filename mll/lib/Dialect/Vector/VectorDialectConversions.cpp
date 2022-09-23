//===- VectorDialectConversions.cpp - -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "mll/Dialect/Vector/VectorDialect.h"

#include "mll/AST/AST.h"
#include "mll/MLIRCodeGen/MLIRCodeGen.h"
#include "mll/Parse/BuiltinDialect.h"
#include "mll/Parse/Parser.h"
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
using namespace mlir;

namespace mll {
namespace vector {

class VecSplatConstConv : public ExprConversion {
  ReturnInfo convertExpr(mlir::OpBuilder &builder, mll::Expr *expr) override {
    auto cExpr = cast<VectorSplatConstantExpr>(expr);
    auto arrTy = cExpr->getType();
    auto vector = (convertType(arrTy)).cast<mlir::VectorType>();
    mlir::Value value = builder.create<mlir::vector::SplatOp>(
        toMLIRLoc(expr), cg->convertExpr(cExpr->value()).getValue(), vector);
    return value;
  }
};

class VecLoadConv : public ExprConversion {
  ReturnInfo convertExpr(mlir::OpBuilder &builder, mll::Expr *expr) override {
    auto vExpr = cast<vector::LoadMethodExpr>(expr);
    auto vecTy = vExpr->getType();
    auto vectorTy = (convertType(vecTy)).cast<mlir::VectorType>();

    auto toIndex = [&](mlir::Value val) {
      return builder.create<arith::IndexCastOp>(toMLIRLoc(expr),
                                                builder.getIndexType(), val);
    };

    auto base = cg->convertExpr(vExpr->arr()).getLHSValue();
    SmallVector<mlir::Value, 4> indices;
    for (auto *index : vExpr->indices()->children()) {
      indices.push_back(toIndex(cg->convertExpr(index).getValue()));
    }
    mlir::Value value = builder.create<mlir::vector::LoadOp>(
        toMLIRLoc(expr), vectorTy, base, indices);
    return value;
  }
};

class ReduceAddConv : public ExprConversion {
  ReturnInfo convertExpr(mlir::OpBuilder &builder, mll::Expr *expr) override {
    auto vExpr = cast<vector::ReduceAddMethodExpr>(expr);
    auto base = cg->convertExpr(vExpr->arg()).getValue();
    mlir::Value value = builder.create<mlir::vector::ReductionOp>(
        toMLIRLoc(expr), mlir::vector::CombiningKind::ADD, base);
    return value;
  }
};

class VectorTypeConversion : public mll::TypeConversion {
  mlir::Type convertType(mlir::OpBuilder &builder, mll::Type *ty) override {
    auto arrTy = llvm::cast<VectorType>(ty);
    return mlir::VectorType::get(arrTy->shape(),
                                 cg->convertType(arrTy->base()));
  }
};

void VectorDialect::registerMLIRConversions(::mll::MLIRCodeGen *cg) {
  cg->registerTypeConversion<VectorType, VectorTypeConversion>();
  cg->registerExprConversion<VectorSplatConstantExpr, VecSplatConstConv>();
  cg->registerExprConversion<LoadMethodExpr, VecLoadConv>();
  cg->registerExprConversion<ReduceAddMethodExpr, ReduceAddConv>();
}

} // namespace vector
} // namespace mll