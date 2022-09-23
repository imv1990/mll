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

#ifndef MLL_INCLUDE_AST_MLLCONTEXT_H
#define MLL_INCLUDE_AST_MLLCONTEXT_H

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mll/AST/BinaryOperator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>

using mlir::TypeID;

namespace mll {

// Represents <dialect-name> and (type |ast-node) name.
typedef std::pair<std::string, std::string> DialectNamePair;

class Dialect;
class Type;
class ASTBuilder;

class MLLContext {
private:
  mutable llvm::BumpPtrAllocator allocator;
  mlir::MLIRContext mlirContext;

  // Stores all unique types built.
  mlir::DenseMap<llvm::hash_code, Type *> typeStorageMap;
  friend class ASTBuilder;

  // Contains all the loaded dialects.
  mlir::DenseMap<TypeID, Dialect *> dialectMap;
  llvm::StringMap<Dialect *> dialectNameMap;

  // Store all the loaded binary operators.
  llvm::StringMap<BinaryOperator> binaryOpMap;

public:
  MLLContext() {
    mlirContext.loadDialect<
        mlir::func::FuncDialect, mlir::memref::MemRefDialect,
        mlir::BuiltinDialect, mlir::arith::ArithmeticDialect,
        mlir::LLVM::LLVMDialect, mlir::bufferization::BufferizationDialect,
        mlir::tosa::TosaDialect, mlir::linalg::LinalgDialect,
        mlir::scf::SCFDialect, mlir::gpu::GPUDialect, mlir::omp::OpenMPDialect,
        mlir::vector::VectorDialect>();
  }

  void *allocate(size_t bytes, size_t align) const {
    return allocator.Allocate(bytes, align);
  }

  void deallocate(void *ptr) const {
    // Automatic de-allocation will happen during destruction of context.
  }

  mlir::MLIRContext *getMLIRContext() { return &mlirContext; }

  template <class DialectT> DialectT *getDialect() {
    auto hashCode = TypeID::get<DialectT>();
    return static_cast<DialectT *>(dialectMap[hashCode]);
  }

  template <class T> void registerDialect() {
    auto val = new (this, alignof(T)) T(this);
    dialectMap[TypeID::get<T>()] = val;
    dialectNameMap[T::getName()] = val;
  }

  Dialect *getDialect(mlir::StringRef name) {
    auto iter = dialectNameMap.find(name);
    if (iter == dialectNameMap.end()) {
      return nullptr;
    }
    return iter->second;
  }

  void registerBinaryOperator(BinaryOperator op);

  BinaryOperator *getBinaryOperator(std::string name) {
    auto val = binaryOpMap.find(name);
    if (val == binaryOpMap.end()) {
      return nullptr;
    }
    return &val->second;
  }

  llvm::SmallVector<Dialect *> getLoadedDialects() {
    llvm::SmallVector<Dialect *> loadedDialects;
    for (auto &dialMap : dialectNameMap) {
      loadedDialects.push_back(dialMap.second);
    }
    return loadedDialects;
  }
};
} // namespace mll

inline void *operator new(size_t Bytes, const mll::MLLContext *context,
                          size_t Alignment = 8) {
  return context->allocate(Bytes, Alignment);
}

inline void operator delete(void *Ptr, const mll::MLLContext *context) {
  context->deallocate(Ptr);
}

#endif