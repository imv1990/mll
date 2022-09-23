//===- DialectGen.cpp - MLL dialect definitions generator -----------------===//
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

#include "Dialect.h"
#include "llvm/Support/CommandLine.h"

static llvm::cl::OptionCategory dialectGenCat("Options for -gen-dialect-*");
static llvm::cl::opt<std::string>
    selectedDialect("dialect-name", llvm::cl::desc("The dialect to gen for"),
                    llvm::cl::cat(dialectGenCat), llvm::cl::CommaSeparated);

/// The code block for the start of a dialect class declaration.
///
/// {0}: The name of the dialect class.
/// {1}: The dialect namespace.
static const char *const dialectDeclBeginStr = R"(

class {0} : public ::mll::Dialect {
  explicit {0}(::mll::MLLContext *context);

  friend class ::mll::MLLContext;
public:
  static std::string getName() {
    return "{1}";
  }

  static bool classof(const ::mll::Dialect *base) {
    return TypeID::get<{0}>() == base->getTypeID();
  }

  void initialize() {{
    registerTypes();
    registerBinaryOperators();
    registerPropertyExprs();
    registerMethodExprs();
    registerConstantExprTypes();
    registerVariableExprTypes();
    registerStmts();
  }

  virtual void registerTypes() override;
  
  virtual void registerStmts() override;

  virtual void registerBinaryOperators() override;

  virtual void registerPropertyExprs() override;

  virtual void registerMethodExprs() override;
  
  virtual void registerConstantExprTypes() override;
  
  virtual void registerVariableExprTypes() override;

  virtual void registerMLIRConversions(::mll::MLIRCodeGen* cg) override;
};
)";

static void emitDialectClass(Dialect &dialect, raw_ostream &os) {

  auto className = dialect.getClassName();
  auto name = dialect.getName();

  NamespaceEmitter emitter(os, {"mll", name});

  os << llvm::formatv(dialectDeclBeginStr, className, name);
}

static bool emitDialectDecls(const RecordKeeper &recordKeeper,
                             raw_ostream &os) {
  emitSourceFileHeader("Dialect Declarations", os);

  std::vector<Record *> defs = recordKeeper.getAllDerivedDefinitions("Dialect");
  assert(!defs.empty());
  Record *dialectDef = nullptr;

  for (auto def : defs) {
    if (Dialect(def).getName() == selectedDialect) {
      dialectDef = def;
      break;
    }
  }

  assert(dialectDef);
  Dialect dialect(dialectDef);
  emitDialectClass(dialect, os);

  auto className = dialect.getClassName();
  auto name = dialect.getName();

  os << "MLIR_DECLARE_EXPLICIT_TYPE_ID("
     << "mll::" << name << "::" << className << ")\n";
  return false;
}

static const char *const dialectConstructorStr = R"(
{0}::{0}(::mll::MLLContext *context) 
    : ::mll::Dialect("{1}", context, ::mlir::TypeID::get<{0}>()) {{
      initialize();
}
)";

static bool emitDialectDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Dialect Definitions", os);

  std::vector<Record *> defs = recordKeeper.getAllDerivedDefinitions("Dialect");
  assert(!defs.empty());
  Record *dialectDef = nullptr;

  for (auto def : defs) {
    if (Dialect(def).getName() == selectedDialect) {
      dialectDef = def;
      break;
    }
  }

  assert(dialectDef);
  Dialect dialect(dialectDef);

  auto className = dialect.getClassName();
  auto name = dialect.getName();

  // Emit the TypeID explicit specializations to have a single symbol def.
  os << "MLIR_DEFINE_EXPLICIT_TYPE_ID("
     << "mll::" << name << "::" << className << ")\n";

  // Emit all nested namespaces.
  NamespaceEmitter emitter(os, {"mll", name});

  os << llvm::formatv(dialectConstructorStr, className, name);
  return false;
}

static mlir::GenRegistration
    genDialectDecls("gen-dialect-decls", "Generate dialect declarations",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      return emitDialectDecls(records, os);
                    });

static mlir::GenRegistration
    genDialectDefs("gen-dialect-defs", "Generate dialect definitions",
                   [](const RecordKeeper &records, raw_ostream &os) {
                     return emitDialectDefs(records, os);
                   });