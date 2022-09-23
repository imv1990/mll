//===- Dialect.h - MLL dialect definitions generator
//-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DialectGen uses the description of dialects to generate C++ definitions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "mlir/TableGen/GenInfo.h"

using namespace llvm;

// A helper RAII class to emit nested namespaces for this op.
class NamespaceEmitter {
public:
  NamespaceEmitter(raw_ostream &os, SmallVector<StringRef, 2> namespaces)
      : os(os), namespaces(namespaces) {
    emitNamespaceStarts(os);
  }
  NamespaceEmitter(raw_ostream &os, StringRef cppNamespace) : os(os) {
    namespaces.push_back(cppNamespace);
    emitNamespaceStarts(os);
  }

  ~NamespaceEmitter() {
    for (StringRef ns : llvm::reverse(namespaces))
      os << "} // namespace " << ns << "\n";
  }

private:
  void emitNamespaceStarts(raw_ostream &os) {
    for (StringRef ns : namespaces)
      os << "namespace " << ns << " {\n";
  }
  raw_ostream &os;
  SmallVector<StringRef, 2> namespaces;
};

class Dialect {
  llvm::Record *def;

public:
  Dialect(llvm::Record *def) : def(def) {}

  std::string getClassName() {
    std::string cppName = def->getName().str();
    llvm::erase_value(cppName, '_');
    return cppName;
  }

  std::string getFullCppClassName() {
    return "::mll::" + getName().str() + "::" + getClassName();
  }

  StringRef getName() { return def->getValueAsString("name"); }
};