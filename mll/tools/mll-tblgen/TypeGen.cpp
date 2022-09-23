//===- TypeGen.cpp - MLL type definitions generator -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TypeGen uses the description of types to generate C++ definitions.
//
//===----------------------------------------------------------------------===//

#include "Type.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/Class.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir::tblgen;

static llvm::cl::OptionCategory dialectGenCat("Options for -gen-types-*");
static llvm::cl::opt<std::string>
    selectedDialect("dialect-name-ty", llvm::cl::desc("The dialect to gen for"),
                    llvm::cl::cat(dialectGenCat), llvm::cl::CommaSeparated);

namespace ns_type {
static void addMethodParameters(Type &type,
                                SmallVector<MethodParameter> &paramList) {
  // Add the remaining arguments.
  ArgumentDAG dag = type.getArguments();
  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto type = dag.getArgType(I);
    bool isVecType = type.isVectorType();
    paramList.emplace_back(type.getTypeStr() + (isVecType ? " &" : ""),
                           dag.getArgName(I));
  }
}

static void declareFieldsAndMembers(Class &typeClass, Type &type) {
  // Add the remaining arguments as field and access functions.
  ArgumentDAG dag = type.getArguments();
  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto type = dag.getArgType(I);
    auto fieldName = dag.getArgFieldName(I);
    auto name = dag.getArgName(I);

    typeClass.addField(type.getTypeStr(), fieldName);

    bool isVecType = type.isVectorType();
    auto method = typeClass.addInlineMethod<Method::Const>(
        (isVecType ? "const " : "") + type.getTypeStr() +
            (isVecType ? " &" : ""),
        name);
    auto &body = method->body().indent();
    body << "return " << fieldName << ";";
  }
}

static void defineConstructor(Class &typeClass, Type &type) {
  SmallVector<mlir::tblgen::MethodParameter> paramList;
  paramList.emplace_back("::mll::MLLContext *", "context");
  addMethodParameters(type, paramList);
  auto *constructor = typeClass.addConstructor<Method::Private>(paramList);

  // Add member initalizer for constructor.
  std::string initString;
  llvm::raw_string_ostream initStr(initString);
  initStr << "context, " << type.getTypeIDExpr();

  initStr << ", std::make_pair(\"" << type.getDialect().getName() << "\", \""
          << type.getName() << "\")";
  initStr.flush();
  constructor->addMemberInitializer("::mll::Type", initString);
  ArgumentDAG dag = type.getArguments();
  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto fieldName = dag.getArgFieldName(I);
    auto name = dag.getArgName(I);
    constructor->addMemberInitializer(fieldName, name);
  }

  ExtraClassDeclaration decl("friend class ::mll::MLLContext;\n");
  ExtraClassDeclaration decl2("friend class ::mll::ASTBuilder;\n");
  typeClass.declare<ExtraClassDeclaration>(decl);
  typeClass.declare<ExtraClassDeclaration>(decl2);
}

static void defineStaticFunctions(Class &typeClass, Type &type) {

  // classof
  auto method = typeClass.addStaticInlineMethod(
      "bool", "classof", MethodParameter("const ::mll::Type* ", "base"));
  auto dummy = " return base->getTypeID() == ";
  method->body().indent() << dummy << type.getTypeIDExpr() << ";";

  // getName()
  method = typeClass.addStaticInlineMethod("std::string", "getName");
  method->body().indent() << "return \"" << type.getName() << "\";";

  // getUniqueName()
  method = typeClass.addStaticInlineMethod("std::string", "getUniqueName");
  method->body().indent() << "return \"" << type.getUniqueName() << "\";";

  // hasCustomParseMethod()
  method = typeClass.addStaticInlineMethod("bool", "hasCustomParseMethod");
  method->body().indent() << "return " << type.needParserMethod() << ";";
}

static const char *const printStringFmt = R"(
mlir::raw_indented_ostream ros(os);
auto &fos = ros.indent();
os << "{0}: " << " {\n";
{1}
fos.unindent();
os << "}\n";
)";
static void defineASTPrintFunction(Class &typeClass, Type &type) {
  auto method = typeClass.addConstMethod(
      "void", "dump", MethodParameter("::llvm::raw_ostream &", "os"));
  method->markOverride();
  auto &body = method->body().indent();

  std::string printStr;
  auto typeName =
      type.getDialect().getName().str() + "." + type.getCppClassName().str();
  ArgumentDAG dag = type.getArguments();

  // Simple print if it is scalar variable.
  if (dag.getNumArgs() == 0) {
    body << "os << \"" << typeName << "\";\n";
    return;
  }

  // Print all fields if it is complex variable.
  llvm::raw_string_ostream print(printStr);
  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto fieldName = dag.getArgFieldName(I);
    std::string value = dag.getArgName(I).str() + "()";
    auto isPtrType = dag.getArgType(I).isASTNodeRelated();
    // De-reference the pointer.
    if (isPtrType) {
      value = "*" + value;
    }
    print << "fos << \"" << dag.getArgName(I) << ": \" << " << value
          << " << \"\\n\";\n";
  }
  print.flush();
  body << llvm::formatv(printStringFmt, typeName, printStr);
}

static void emitTypeClass(Type &type, raw_ostream &os, bool isDef = false) {

  auto className = type.getCppClassName();
  auto dialectName = type.getDialect().getName();
  auto fullClassName = type.getFullClassName();

  ::NamespaceEmitter emitter(os, {"mll", dialectName});

  mlir::tblgen::Class typeClass(className);
  typeClass.addParent(std::move(ParentClass("::mll::Type")));

  declareFieldsAndMembers(typeClass, type);

  defineConstructor(typeClass, type);

  // Add static functions
  defineStaticFunctions(typeClass, type);

  // Add parser declaration
  if (type.needParserMethod()) {
    typeClass.declareStaticMethod(
        type.getFullClassName() + "* ", "parse",
        MethodParameter("::mll::Parser &", "parser"),
        MethodParameter("::mll::ASTBuilder &", "builder"));
  } else {
    // FIXME: Accept from user?
    auto method = typeClass.addStaticMethod(
        type.getFullClassName() + "* ", "parse",
        MethodParameter("::mll::Parser &", "parser"),
        MethodParameter("::mll::ASTBuilder &", "builder"));
    method->body().indent() << "return nullptr;";
  }

  // Add dump method
  defineASTPrintFunction(typeClass, type);

  typeClass.finalize();

  if (!isDef) {
    typeClass.writeDeclTo(os);
  } else {
    typeClass.writeDefTo(os);
  }
}

static bool emitTypeDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Type Declarations", os);
  std::vector<Record *> defs = recordKeeper.getAllDerivedDefinitions("Type");
  for (auto &def : defs) {
    Type type(def);
    if (type.getDialect().getName() != selectedDialect) {
      continue;
    }
    emitTypeClass(type, os);
    os << "MLIR_DECLARE_EXPLICIT_TYPE_ID(" << type.getFullClassName() << ")\n";
  }
  return false;
}

static ::Dialect getSingleDialect(const RecordKeeper &recordKeeper) {
  std::vector<Record *> defs = recordKeeper.getAllDerivedDefinitions("Dialect");
  for (auto &def : defs) {
    if ((::Dialect(def)).getName() == selectedDialect) {
      return ::Dialect(def);
    }
  }
  llvm_unreachable("Dialect definition not found");
}

static bool emitTypeDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Type Definitions", os);

  std::vector<Record *> defs = recordKeeper.getAllDerivedDefinitions("Type");
  std::vector<std::string> typeList;
  std::string fullDialectName =
      getSingleDialect(recordKeeper).getFullCppClassName();

  for (auto &def : defs) {
    Type type(def);

    if (type.getDialect().getName() != selectedDialect) {
      continue;
    }

    // Emit the TypeID explicit specializations to have a single symbol
    os << "MLIR_DEFINE_EXPLICIT_TYPE_ID(" << type.getFullClassName() << ")\n";

    emitTypeClass(type, os, true);

    typeList.push_back(type.getFullClassName());
  }

  auto funcName = fullDialectName + "::registerTypes";
  auto method =
      mlir::tblgen::Method("void", funcName, Method::Properties::None);
  auto &body = method.body().indent();
  for (auto &type : typeList) {
    body << "registerType<" << type << ">();\n";
  }

  mlir::raw_indented_ostream fos(os);
  method.writeDefTo(fos, "");

  return false;
}

} // namespace ns_type

static mlir::GenRegistration
    genTypeDecls("gen-type-decls", "Generate type declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   return ns_type::emitTypeDecls(records, os);
                 });

static mlir::GenRegistration
    genTypeDefs("gen-type-defs", "Generate type definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  return ns_type::emitTypeDefs(records, os);
                });