//===- BuiltinDialect.cpp - -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef MLL_PARSE_BUILTIN_DIALECT_H
#define MLL_PARSE_BUILTIN_DIALECT_H

#include "mll/AST/Dialect.h"
#include "mll/AST/MLLContext.h"
#include "mll/AST/Type.h"
#include "mll/Parse/Parser.h"

#include "mll/Parse/BuiltinDialect.h.inc"

#include "mll/Parse/BuiltinDialectTypes.h.inc"

#include "mll/Parse/BuiltinDialectNodes.h.inc"

#include "llvm/Support/raw_ostream.h"

namespace mll {
namespace builtin {
class ListExpr final : public Expr {
  explicit ListExpr(Location loc, builtin::ListType *ty, ExprListRef nodes);
  friend class ::mll::ASTBuilder;
  friend class ::mll::MLLContext;

public:
  Type *getType() const { return getExprType(); }

  static bool classof(const ::mll::Expr *base) {
    return base->getTypeID() == ::mlir::TypeID::get<::mll::builtin::ListExpr>();
  }

  void dump(llvm::raw_ostream &os) const override;
};
} // namespace builtin
} // namespace mll

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mll::builtin::ListExpr)

OVERLOAD_OSTREAM_OPERATOR(mll::builtin::ListExpr)

template <class T>
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const llvm::SmallVectorImpl<T> &val) {
  os << "[";
  for (unsigned I = 0; I < val.size(); ++I) {
    if (I != 0) {
      os << ", ";
    }
    os << val[I];
  }
  os << "]";

  return os;
}

#endif