//===- MLIRCodeGen.h - MLL Parser Interface ---------------------*- C++--*-===//
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

#include "mlir/IR/Location.h"
#include "mll/AST/AST.h"
#include "mll/AST/MLLContext.h"
#include "mll/MLIRCodeGen/MLIRCodeGen.h"
#include "mll/Parse/BuiltinDialect.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/IR/Function.h"
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/Bufferize.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <string>

using namespace mll;
using namespace mll::builtin;

using namespace mlir;

class I32TypeConversion : public mll::TypeConversion {

  mlir::Type convertType(mlir::OpBuilder &builder, mll::Type *ty) override {
    return mlir::IntegerType::get(builder.getContext(), 32);
  }
};

class I1TypeConversion : public mll::TypeConversion {

  mlir::Type convertType(mlir::OpBuilder &builder, mll::Type *ty) override {
    return mlir::IntegerType::get(builder.getContext(), 1);
  }
};

class F32TypeConversion : public mll::TypeConversion {

  mlir::Type convertType(mlir::OpBuilder &builder, mll::Type *ty) override {
    return builder.getF32Type();
  }
};

class ArrayTypeConversion : public mll::TypeConversion {
  mlir::Type convertType(mlir::OpBuilder &builder, mll::Type *ty) override {
    auto arrTy = llvm::cast<ArrayType>(ty);
    return MemRefType::get(arrTy->shape(), cg->convertType(arrTy->base()));
  }
};

class FunctionTypeConversion : public mll::TypeConversion {
  mlir::Type convertType(mlir::OpBuilder &builder, mll::Type *ty) override {
    auto funcTy = llvm::cast<builtin::FunctionType>(ty);

    llvm::SmallVector<mlir::Type> rets, ins;

    if (!llvm::isa<builtin::NoneType>(funcTy->returnType())) {
      rets.push_back(cg->convertType(funcTy->returnType()));
    }

    for (auto &arg : funcTy->args()) {
      ins.push_back(cg->convertType(arg));
    }
    return builder.getFunctionType(ins, rets);
  }
};

class AssignStmtConversion : public mll::StmtConversion {
  void convertStmt(mlir::OpBuilder &builder, mll::Stmt *stmt) override {
    auto assignStmt = llvm::cast<mll::builtin::AssignStmt>(stmt);

    auto loc = toMLIRLoc(stmt);
    auto lhs = convertExpr(assignStmt->lhs());
    auto rhs = convertExpr(assignStmt->rhs());

    if (lhs.getValue().getType().isa<TensorType>()) {
      auto memref = lhs.getLHSValue();
      auto rhsVal = builder.create<bufferization::ToMemrefOp>(
          loc, memref.getType(), rhs.getValue());
      builder.create<mlir::memref::CopyOp>(loc, rhsVal, memref);
    } else {
      auto loadOp =
          mlir::cast<mlir::memref::LoadOp>(lhs.getValue().getDefiningOp());
      auto memref = loadOp.getMemRef();
      auto indices = loadOp.indices();
      loadOp->erase();
      builder.create<mlir::memref::StoreOp>(loc, rhs.getValue(), memref,
                                            indices);
    }
  }
};

class SymbolExprConv : public ExprConversion {
  ReturnInfo convertExpr(mlir::OpBuilder &builder, mll::Expr *expr) override {
    auto symbolExpr = cast<SymbolExpr>(expr);
    auto loc = toMLIRLoc(expr);
    auto varName = symbolExpr->symbol()->getName();

    if (isa<ArrayType>(symbolExpr->getType())) {
      return builder
          .create<bufferization::ToTensorOp>(loc, cg->getAllocaFor(varName))
          .getResult();
    }

    auto var = cg->getAllocaFor(varName);
    if (symbolExpr->symbol()->isLoopIndVar()) {
      return var;
    }
    mlir::Value loadVal = builder.create<mlir::memref::LoadOp>(
        toMLIRLoc(expr), cg->getAllocaFor(varName));

    return ReturnInfo(loadVal);
  }
};

class ArrAccessConv : public ExprConversion {
  ReturnInfo convertExpr(mlir::OpBuilder &builder, mll::Expr *expr) override {
    auto arrAcc = cast<ArrayAccessVariableExpr>(expr);
    auto loc = toMLIRLoc(expr);
    auto varName = arrAcc->base()->getName();
    auto indices = arrAcc->indices()->children();

    SmallVector<mlir::Value, 8> mlirIndices;

    auto castToIndex = [&](mlir::Value val) {
      return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                                val);
    };

    for (auto &index : indices) {
      auto val = cg->convertExpr(index).getValue();
      mlirIndices.push_back(castToIndex(val));
    }

    mlir::Value loadVal = builder.create<mlir::memref::LoadOp>(
        loc, cg->getAllocaFor(varName), mlirIndices);

    return loadVal;
  }
};

template <class T>
static mlir::Value emitBinaryOp(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value lhs, mlir::Value rhs) {
  return builder.create<T>(loc, lhs.getType(), lhs, rhs);
}

template <arith::CmpIPredicate predicate>
static mlir::Value emitICmpOp(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value lhs, mlir::Value rhs) {
  return builder.create<arith::CmpIOp>(loc, predicate, lhs, rhs);
}

template <arith::CmpFPredicate predicate>
static mlir::Value emitFCmpOp(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value lhs, mlir::Value rhs) {
  return builder.create<arith::CmpFOp>(loc, predicate, lhs, rhs);
}

class BinaryOpConv : public ExprConversion {

  bool isIntegerBasedType(mlir::Type ty) {
    if (ty.isIntOrIndex()) {
      return true;
    }

    if (auto shapedTy = ty.dyn_cast_or_null<ShapedType>()) {
      return (isIntegerBasedType(shapedTy.getElementType()));
    }

    return false;
  }

  bool isFloatingType(mlir::Type ty) {
    if (ty.isa<FloatType>()) {
      return true;
    }

    if (auto shapedTy = ty.dyn_cast_or_null<ShapedType>()) {
      return (isFloatingType(shapedTy.getElementType()));
    }

    return false;
  }

  ReturnInfo convertExpr(mlir::OpBuilder &builder, mll::Expr *expr) override {
    auto binExpr = cast<BinaryOpExpr>(expr);

    auto lhsVal = cg->convertExpr(binExpr->lhs());
    auto rhsVal = cg->convertExpr(binExpr->rhs());

    auto lhs = lhsVal.getValue();
    auto rhs = rhsVal.getValue();
    auto loc = toMLIRLoc(expr);
    auto ty = lhs.getType();

    typedef llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Location,
                                           mlir::Value, mlir::Value)>
        binaryOpFn;

    if (lhs.getType().isa<TensorType>()) {
      auto fnPtr = StringSwitch<binaryOpFn>(binExpr->op())
                       .Case("+", emitBinaryOp<tosa::AddOp>)
                       .Case("-", emitBinaryOp<tosa::SubOp>)
                       .Case("/", emitBinaryOp<tosa::DivOp>)
                       .Default(nullptr);
      assert(fnPtr);

      if (binExpr->op() == "*") {
        return builder
            .create<tosa::MulOp>(loc, ty, lhs, rhs,
                                 builder.getI32IntegerAttr(0))
            .getResult();
      }
      return fnPtr(builder, loc, lhs, rhs);
    }

    if (isIntegerBasedType(ty)) {

      auto fnPtr = StringSwitch<binaryOpFn>(binExpr->op())
                       .Case("+", emitBinaryOp<arith::AddIOp>)
                       .Case("-", emitBinaryOp<arith::SubIOp>)
                       .Case("*", emitBinaryOp<arith::MulIOp>)
                       .Case("/", emitBinaryOp<arith::DivSIOp>)
                       .Case("<", emitICmpOp<arith::CmpIPredicate::slt>)
                       .Case(">", emitICmpOp<arith::CmpIPredicate::sgt>)
                       .Case("<=", emitICmpOp<arith::CmpIPredicate::sle>)
                       .Case(">=", emitICmpOp<arith::CmpIPredicate::sge>)
                       .Case("==", emitICmpOp<arith::CmpIPredicate::eq>)
                       .Case("!=", emitICmpOp<arith::CmpIPredicate::ne>)
                       .Case("and", emitBinaryOp<arith::AndIOp>)
                       .Case("or", emitBinaryOp<arith::OrIOp>)
                       .Default(nullptr);
      assert(fnPtr);
      return fnPtr(builder, loc, lhs, rhs);
    }
    // Check if it is float type.
    assert(isFloatingType(ty));

    auto fnPtr = StringSwitch<binaryOpFn>(binExpr->op())
                     .Case("+", emitBinaryOp<arith::AddFOp>)
                     .Case("-", emitBinaryOp<arith::SubFOp>)
                     .Case("*", emitBinaryOp<arith::MulFOp>)
                     .Case("/", emitBinaryOp<arith::DivFOp>)
                     .Case("<", emitFCmpOp<arith::CmpFPredicate::OLT>)
                     .Case(">", emitFCmpOp<arith::CmpFPredicate::OGT>)
                     .Case("<=", emitFCmpOp<arith::CmpFPredicate::OLE>)
                     .Case(">=", emitFCmpOp<arith::CmpFPredicate::OGE>)
                     .Case("==", emitFCmpOp<arith::CmpFPredicate::OEQ>)
                     .Case("!=", emitFCmpOp<arith::CmpFPredicate::ONE>)
                     .Default(nullptr);
    assert(fnPtr);
    return fnPtr(builder, loc, lhs, rhs);
  }
};

class I32ConstConv : public ExprConversion {
  ReturnInfo convertExpr(mlir::OpBuilder &builder, mll::Expr *expr) override {
    auto i32ConstExpr = cast<I32ConstantExpr>(expr);
    mlir::Value val = builder.create<arith::ConstantIntOp>(
        toMLIRLoc(expr), i32ConstExpr->value(), builder.getI32Type());

    return val;
  }
};

class F32ConstConv : public ExprConversion {
  ReturnInfo convertExpr(mlir::OpBuilder &builder, mll::Expr *expr) override {
    auto f32ConstExpr = cast<F32ConstantExpr>(expr);

    auto floatAttr = builder.getF32FloatAttr(f32ConstExpr->value());
    mlir::Value val = builder.create<arith::ConstantOp>(
        toMLIRLoc(expr), floatAttr, builder.getF32Type());

    return val;
  }
};

class DenseArrayConstConv : public ExprConversion {
  ReturnInfo convertExpr(mlir::OpBuilder &builder, mll::Expr *expr) override {
    auto arrCExpr = cast<DenseArrayConstantExpr>(expr);
    auto arrTy = arrCExpr->getType();
    auto baseTy = arrTy->base();

    auto memref = (convertType(arrTy)).cast<MemRefType>();
    auto tensor =
        RankedTensorType::get(memref.getShape(), memref.getElementType());

    /// Check if it is dense operation.
    if (isa<I32Type>(baseTy) && isa<I32ConstantExpr>(arrCExpr->value())) {
      auto i32Const = cast<I32ConstantExpr>(arrCExpr->value());
      auto constInt = DenseElementsAttr::get(tensor, i32Const->value());
      mlir::Value value =
          builder.create<arith::ConstantOp>(toMLIRLoc(expr), constInt, tensor);
      return value;
    }

    /// Check if it is dense operation.
    if (isa<F32Type>(baseTy) && isa<F32ConstantExpr>(arrCExpr->value())) {
      auto f32Const = cast<F32ConstantExpr>(arrCExpr->value());
      auto constFloat = DenseElementsAttr::get(tensor, f32Const->value());
      mlir::Value value = builder.create<arith::ConstantOp>(toMLIRLoc(expr),
                                                            constFloat, tensor);
      return value;
    }

    assert(false && "unhandled array constant expression");
  }
};

class PrintStmtConv : public StmtConversion {

  LLVM::LLVMFuncOp getOrInsertPrintFunc(mlir::OpBuilder &builder,
                                        StringRef funcName,
                                        ArrayRef<mlir::Type> argTypes) {
    auto module = cg->getModule();
    if (auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
      return funcOp;
    }

    auto llvmFnType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(builder.getContext()), argTypes);

    OpBuilder::InsertionGuard gaurd(builder);
    builder.setInsertionPointToStart(module.getBody());
    return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName,
                                            llvmFnType);
  }

  void printArray(mlir::OpBuilder &builder, mlir::Value arg) {

    std::string funcName = "printMemref";
    auto tensorTy = arg.getType().cast<TensorType>();

    auto memrefTy =
        MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
    auto umemRefTy = UnrankedMemRefType::get(tensorTy.getElementType(),
                                             builder.getI32IntegerAttr(0));
    assert(tensorTy.getElementType().isIntOrIndexOrFloat());
    std::string typeName =
        tensorTy.getElementType().isInteger(32) ? "I32" : "F32";
    funcName = funcName + typeName;

    auto module = cg->getModule();
    func::FuncOp funcOp = module.lookupSymbol<mlir::func::FuncOp>(funcName);
    if (!funcOp) {
      auto funcType = builder.getFunctionType(umemRefTy, llvm::None);
      OpBuilder::InsertionGuard gaurd(builder);
      builder.setInsertionPointToStart(module.getBody());
      funcOp =
          builder.create<func::FuncOp>(module.getLoc(), funcName, funcType);
      funcOp.setSymVisibilityAttr(builder.getStringAttr("private"));
    }

    arg =
        builder.create<bufferization::ToMemrefOp>(arg.getLoc(), memrefTy, arg);
    auto cast =
        builder.create<mlir::memref::CastOp>(arg.getLoc(), umemRefTy, arg);
    builder.create<mlir::func::CallOp>(arg.getLoc(), funcOp,
                                       ArrayRef<Value>{cast});
  }

  void printVector(OpBuilder &builder, mlir::Value val) {
    builder.create<mlir::vector::PrintOp>(val.getLoc(), val);
  }

  void createPrintNewLineFunc(OpBuilder &builder, mlir::Location loc) {
    auto funcOp = getOrInsertPrintFunc(builder, "printNewLine", llvm::None);
    builder.create<LLVM::CallOp>(loc, funcOp, ArrayRef<Value>{});
  }

  void createPrintInt(OpBuilder &builder, mlir::Location loc, mlir::Value val) {
    auto funcOp =
        getOrInsertPrintFunc(builder, "print_i32", {builder.getI32Type()});
    builder.create<LLVM::CallOp>(loc, funcOp, ArrayRef<Value>{val});
  }

  void createPrintBool(OpBuilder &builder, mlir::Location loc,
                       mlir::Value val) {
    auto funcOp =
        getOrInsertPrintFunc(builder, "print_i1", {builder.getI1Type()});
    builder.create<LLVM::CallOp>(loc, funcOp, ArrayRef<Value>{val});
  }

  void createPrintFloat(OpBuilder &builder, mlir::Location loc,
                        mlir::Value val) {
    auto funcOp =
        getOrInsertPrintFunc(builder, "print_f32", {builder.getF32Type()});
    builder.create<LLVM::CallOp>(loc, funcOp, ArrayRef<Value>{val});
  }

  void createPrintSpace(OpBuilder &builder, mlir::Location loc) {
    auto funcOp = getOrInsertPrintFunc(builder, "printSpace", {});
    builder.create<LLVM::CallOp>(loc, funcOp, ArrayRef<Value>{});
  }

  void createPrintForString(OpBuilder &builder, mlir::Location loc,
                            StringRef name, std::string value) {

    auto str = mlir::LLVM::createGlobalString(
        loc, builder, name, value, mlir::LLVM::linkage::Linkage::Internal);

    auto llvmI8PtrTy = LLVM::LLVMPointerType::get(builder.getI8Type());
    auto funcOp = getOrInsertPrintFunc(builder, "printStr", {llvmI8PtrTy});
    builder.create<LLVM::CallOp>(loc, funcOp, ArrayRef<Value>{str});
  }

  void convertStmt(mlir::OpBuilder &builder, mll::Stmt *stmt) override {
    auto printStmt = cast<PrintStmt>(stmt);
    auto &exprList = printStmt->args()->children();

    std::string name = "print_" + std::to_string(stmt->getLoc().lineNo) + "_";
    unsigned argNum = 0;
    for (auto &expr : exprList) {
      if (auto strExpr = dyn_cast<StringConstantExpr>(expr)) {
        std::string val = strExpr->value();
        val = val + '\0';
        auto currName = name + std::to_string(argNum++);
        createPrintForString(builder, toMLIRLoc(expr), currName, val);
        createPrintSpace(builder, toMLIRLoc(expr));
        continue;
      }

      if (llvm::isa<I32Type>(expr->getExprType())) {
        createPrintInt(builder, toMLIRLoc(expr), convertExpr(expr).getValue());
        continue;
      }

      if (llvm::isa<I1Type>(expr->getExprType())) {
        createPrintBool(builder, toMLIRLoc(expr), convertExpr(expr).getValue());
        continue;
      }

      if (llvm::isa<F32Type>(expr->getExprType())) {
        createPrintFloat(builder, toMLIRLoc(expr),
                         convertExpr(expr).getValue());
        continue;
      }

      if (llvm::isa<ArrayType>(expr->getExprType())) {
        printArray(builder, convertExpr(expr).getValue());
        continue;
      }

      auto mlirTy = convertType(expr->getExprType());
      if (mlirTy.isa<VectorType>()) {
        printVector(builder, convertExpr(expr).getValue());
        continue;
      }

      assert(false && "Unhandled type in print");
    }

    createPrintNewLineFunc(builder, toMLIRLoc(stmt));
  }
};

template <class ExprT, class OpT>
class UnaryMethodExpr : public ExprConversion {
  ReturnInfo convertExpr(mlir::OpBuilder &builder, mll::Expr *expr) override {
    auto exp = cast<ExprT>(expr);
    auto arg = cg->convertExpr(exp->arg()).getValue();
    return builder.create<OpT>(toMLIRLoc(expr), arg.getType(), arg).getResult();
  }
};

template <class ExprT, class OpT>
class RedUnaryMethodExpr : public ExprConversion {
  ReturnInfo convertExpr(mlir::OpBuilder &builder, mll::Expr *expr) override {
    auto exp = cast<ExprT>(expr);
    auto arg = cg->convertExpr(exp->arg()).getValue();
    auto tensor = arg.getType().template cast<RankedTensorType>();
    auto outType = RankedTensorType::get({1l}, tensor.getElementType());
    return builder
        .create<OpT>(toMLIRLoc(expr), outType, arg,
                     builder.getI64IntegerAttr(0))
        .getResult();
  }
};

class ForStmtConv : public StmtConversion {
  void convertStmt(mlir::OpBuilder &builder, mll::Stmt *stmt) override {
    auto forStmt = cast<ForStmt>(stmt);
    auto loc = toMLIRLoc(stmt);

    auto toIndex = [&](mlir::Value val) {
      return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                                val);
    };

    auto toI32 = [&](mlir::Value val) {
      return builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), val);
    };

    auto lb = cg->convertExpr(forStmt->lb()).getValue();
    auto ub = cg->convertExpr(forStmt->ub()).getValue();
    auto step = cg->convertExpr(forStmt->step()).getValue();
    auto scfForOp = builder.create<scf::ForOp>(loc, toIndex(lb), toIndex(ub),
                                               toIndex(step));

    auto body = scfForOp.getBody();
    auto iv = scfForOp.getInductionVar();

    MLIRCodeGen::Scope scope(forStmt->body(), forStmt->symTable(), body);
    cg->pushScope(scope);

    builder.setInsertionPointToEnd(body);
    cg->setAllocaFor(forStmt->iv()->getName(), toI32(iv));
    cg->convertBlock(false);

    // Move yield as last instruction.
    body->front().moveBefore(body, body->end());
    cg->popScope();
  }
};

class IfStmtConv : public StmtConversion {
  void convertStmt(mlir::OpBuilder &builder, mll::Stmt *stmt) override {
    auto ifStmt = cast<IfStmt>(stmt);

    auto conditions = ifStmt->conditions()->children();
    auto blocks = ifStmt->blocksList();
    auto symTables = ifStmt->symTables();

    unsigned numScopes = 0;

    for (unsigned I = 0; I < conditions.size(); ++I) {

      auto loc = toMLIRLoc(conditions[I]->getLoc());
      auto mlirCond = convertExpr(conditions[I]).getValue();
      auto scfIfOp =
          builder.create<scf::IfOp>(loc, mlirCond, blocks.size() > I + 1);

      auto ifBody = scfIfOp.thenBlock();

      MLIRCodeGen::Scope scope(blocks[I], symTables[I], ifBody);
      cg->pushScope(scope);
      cg->convertBlock(false);
      // Move yield as last instruction.
      ifBody->front().moveBefore(ifBody, ifBody->end());
      cg->popScope();

      if (blocks.size() > I + 1) {
        auto elseBody = scfIfOp.elseBlock();
        MLIRCodeGen::Scope scope(blocks[I + 1], symTables[I + 1], elseBody);
        cg->pushScope(scope);
        numScopes++;
      }
    }

    if (numScopes > 0) {
      cg->convertBlock(false);
      while (numScopes != 0) {
        auto scope = cg->popScope();
        // Move yield as last instruction.
        scope.mlirBlock->front().moveBefore(scope.mlirBlock,
                                            scope.mlirBlock->end());
        numScopes--;
      }
    }
  }
};

class FuncStmtConv : public StmtConversion {

  void convertStmt(mlir::OpBuilder &builder, mll::Stmt *stmt) override {
    auto funcStmt = cast<FuncStmt>(stmt);
    auto loc = toMLIRLoc(stmt);

    mlir::func::FuncOp funcOp;
    auto funcType =
        convertType(funcStmt->funcType()).cast<mlir::FunctionType>();
    funcOp = mlir::func::FuncOp::create(loc, funcStmt->name(), funcType);
    cg->getModule().insert(cg->getFuncLikeOpBodyInStack()->getParentOp(),
                           funcOp);

    auto block = funcOp.addEntryBlock();
    MLIRCodeGen::Scope scope(funcStmt->body(), funcStmt->symTable(), block,
                             block);
    cg->pushScope(scope);

    builder.setInsertionPointToEnd(block);
    for (unsigned I = 0; I < funcStmt->args().size(); ++I) {
      auto sym = funcStmt->args()[I];
      cg->setAllocaFor(sym->getName(), block->getArgument(I));
    }
    cg->convertBlock(false);

    if (block->empty() || !isa<mlir::func::ReturnOp>(block->back())) {
      builder.create<mlir::func::ReturnOp>(
          block->empty() ? loc : block->back().getLoc());
    }

    cg->popScope();
  }
};

class ExprStmtConv : public StmtConversion {

  void convertStmt(mlir::OpBuilder &builder, mll::Stmt *stmt) override {
    auto exprStmt = cast<ExprStmt>(stmt);
    cg->convertExpr(exprStmt->arg());
  }
};

class CallExprConv : public ExprConversion {
  ReturnInfo convertExpr(mlir::OpBuilder &builder, mll::Expr *expr) override {
    auto callExpr = cast<CallExpr>(expr);

    llvm::SmallVector<mlir::Value> values;
    for (auto arg : callExpr->args()->children()) {

      auto val = cg->convertExpr(arg);
      auto finalVal = val.getValue().getType().isa<TensorType>()
                          ? val.getLHSValue()
                          : val.getValue();
      values.push_back(finalVal);
    }

    auto funcOp = cg->getModule().lookupSymbol<mlir::func::FuncOp>(
        callExpr->func()->getName());
    assert(funcOp);
    auto callOp =
        builder.create<mlir::func::CallOp>(toMLIRLoc(expr), funcOp, values);

    assert(callOp->getNumResults() <= 1);
    return callOp->getResult(0);
  }
};

class ReturnStmtConv : public StmtConversion {
  void convertStmt(mlir::OpBuilder &builder, mll::Stmt *stmt) override {
    auto retStmt = cast<ReturnStmt>(stmt);

    llvm::SmallVector<mlir::Value> operands;
    if (!llvm::isa<builtin::NoneType>(retStmt->arg()->getExprType())) {
      operands.push_back(convertExpr(retStmt->arg()).getValue());
    }
    builder.create<mlir::func::ReturnOp>(toMLIRLoc(stmt), operands);
  }
};

void mll::builtin::BuiltinDialect::registerMLIRConversions(MLIRCodeGen *cg) {

  cg->registerTypeConversion<mll::builtin::I32Type, I32TypeConversion>();
  cg->registerTypeConversion<mll::builtin::I1Type, I1TypeConversion>();
  cg->registerTypeConversion<mll::builtin::F32Type, F32TypeConversion>();
  cg->registerTypeConversion<ArrayType, ArrayTypeConversion>();
  cg->registerTypeConversion<FunctionType, FunctionTypeConversion>();

  cg->registerStmtConversion<mll::builtin::AssignStmt, AssignStmtConversion>();

  cg->registerExprConversion<SymbolExpr, SymbolExprConv>();
  cg->registerExprConversion<BinaryOpExpr, BinaryOpConv>();
  cg->registerExprConversion<ArrayAccessVariableExpr, ArrAccessConv>();
  cg->registerExprConversion<CallExpr, CallExprConv>();

  cg->registerExprConversion<ExpMethodExpr,
                             UnaryMethodExpr<ExpMethodExpr, tosa::ExpOp>>();
  cg->registerExprConversion<
      SumMethodExpr, RedUnaryMethodExpr<SumMethodExpr, tosa::ReduceSumOp>>();
  cg->registerExprConversion<
      MaxMethodExpr, RedUnaryMethodExpr<MaxMethodExpr, tosa::ReduceMaxOp>>();

  cg->registerExprConversion<I32ConstantExpr, I32ConstConv>();
  cg->registerExprConversion<F32ConstantExpr, F32ConstConv>();
  cg->registerExprConversion<DenseArrayConstantExpr, DenseArrayConstConv>();

  cg->registerStmtConversion<PrintStmt, PrintStmtConv>();
  cg->registerStmtConversion<ForStmt, ForStmtConv>();
  cg->registerStmtConversion<IfStmt, IfStmtConv>();
  cg->registerStmtConversion<FuncStmt, FuncStmtConv>();
  cg->registerStmtConversion<ExprStmt, ExprStmtConv>();
  cg->registerStmtConversion<ReturnStmt, ReturnStmtConv>();
}