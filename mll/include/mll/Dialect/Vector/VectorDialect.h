//===- VectorDialect.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef MLL_PARSE_Vector_DIALECT_H
#define MLL_PARSE_Vector_DIALECT_H

#include "mll/AST/AST.h"
#include "mll/AST/Dialect.h"
#include "mll/AST/MLLContext.h"
#include "mll/AST/Type.h"
#include "mll/Parse/BuiltinDialect.h"
#include "mll/Parse/Parser.h"

#include "mll/Dialect/Vector/VectorDialect.h.inc"

#include "mll/Dialect/Vector/VectorDialectTypes.h.inc"

#include "mll/Dialect/Vector/VectorDialectNodes.h.inc"

#include "llvm/Support/raw_ostream.h"

#endif