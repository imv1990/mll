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

#ifndef MLL_INCLUDE_AST_MLLTYPE_H
#define MLL_INCLUDE_AST_MLLTYPE_H

#include "mll/AST/MLLContext.h"
#include "llvm/Support/raw_ostream.h"

using mlir::TypeID;

namespace mll {

class Type {
  MLLContext *context;
  TypeID id;
  DialectNamePair pair;

public:
  TypeID getTypeID() const { return id; }

  virtual void dump(llvm::raw_ostream &os) const = 0;

protected:
  Type(MLLContext *context, TypeID id, DialectNamePair pair)
      : context(context), id(id), pair(pair) {}

public:
  Dialect *getDialect(MLLContext *ctx) const {
    auto d = ctx->getDialect(pair.first);
    assert(d);
    return d;
  }

  mlir::StringRef getName() const { return pair.second; }

  std::string getUniqueName() const { return pair.first + "." + pair.second; }

  MLLContext *getContext() { return context; }

  virtual ~Type() {}
};

} // namespace mll

namespace llvm {

template <class T>
inline llvm::hash_code hash_value(llvm::SmallVector<T> vals) {
  llvm::hash_code code = llvm::hash_value("SmallVector<int>");

  for (auto &val : vals) {
    code = llvm::hash_combine(val, code);
  }
  return code;
}
} // namespace llvm
#endif
