//===- ExprGen.cpp - MLL type definitions generator -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ExprGen uses the description of types to generate C++ definitions.
//
//===----------------------------------------------------------------------===//

#include "ASTNodes.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <mlir/TableGen/CodeGenHelpers.h>
#include <string>

using namespace mlir::tblgen;

static llvm::cl::OptionCategory dialectGenCat("Options for -gen-nodes-*");
static llvm::cl::opt<std::string>
    selectedDialect("dialect-name-node",
                    llvm::cl::desc("The dialect to gen for"),
                    llvm::cl::cat(dialectGenCat), llvm::cl::CommaSeparated);

namespace ns_expr {
static void addMethodParameters(Expr &expr,
                                SmallVector<MethodParameter> &paramList) {
  // Add return type to the constructor
  auto returnType = expr.getReturnType();
  paramList.emplace_back(returnType.getTypeStr(), StringRef("type"));

  if (expr.isBuiltinMethod()) {
    // Add the remaining arguments.
    ArgumentDAG dag = expr.getMethodArguments();
    for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
      std::string typeStr = "";
      if (dag.isTypeConstraint(I)) {
        typeStr = "::mll::Expr*";
      } else {
        auto type = dag.getArgType(I);
        typeStr = type.getTypeStr();
      }
      paramList.emplace_back(typeStr, dag.getArgName(I));
    }
    return;
  }

  // Add the remaining arguments.
  ArgumentDAG dag = expr.getArguments();
  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto type = dag.getArgType(I);
    auto typeStr = type.getTypeStr();
    paramList.emplace_back(typeStr, dag.getArgName(I));
  }
}

static void declareFieldsAndMembersForMethodExpr(Class &exprClass, Expr &expr) {
  auto dag = expr.getMethodArguments();

  unsigned exprNum = 0;
  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto argName = dag.getArgName(I);

    std::string funcRetType, returnExpr;
    if (dag.isTypeConstraint(I)) {
      funcRetType = "::mll::Expr* ";
      returnExpr = "getExpr(" + std::to_string(exprNum++) + ")";
    }

    else if (dag.getArgType(I).isTypeArgument()) {
      funcRetType = dag.getArgType(I).getTypeStr();
      returnExpr = dag.getArgFieldName(I);
      exprClass.addField(funcRetType, returnExpr);
    } else {
      llvm::PrintFatalError(expr.getLoc(), "Wrong argument " +
                                               std::to_string(I) +
                                               " for MethodExpr");
    }

    auto method =
        exprClass.addInlineMethod<Method::Const>(funcRetType, argName);
    method->body().indent() << "return " << returnExpr << ";\n";
  }
}

static void declareFieldsAndMembers(Class &exprClass, Expr &expr) {
  auto returnType = expr.getReturnType();
  auto method = exprClass.addInlineMethod(returnType.getTypeStr(), "getType");
  auto &body = method->body().indent();

  body << "return static_cast<" << returnType.getTypeStr()
       << ">(getExprType());";

  if (expr.isBuiltinMethod()) {
    declareFieldsAndMembersForMethodExpr(exprClass, expr);
    return;
  }

  // Add the remaining arguments as field and access functions.
  ArgumentDAG dag = expr.getArguments();
  for (unsigned I = 0; I < dag.others.size(); ++I) {
    auto type = dag.others[I].type;
    auto fieldName = dag.getArgFieldName(dag.others[I].index);
    auto name = dag.others[I].name;
    exprClass.addField(type.getTypeStr(), fieldName);

    auto method =
        exprClass.addInlineMethod<Method::Const>(type.getTypeStr(), name);
    auto &body = method->body().indent();
    body << "return " << fieldName << ";";
  }

  for (unsigned I = 0; I < dag.expressions.size(); ++I) {
    auto type = dag.expressions[I].type;
    auto name = dag.expressions[I].name;
    auto method =
        exprClass.addInlineMethod<Method::Const>(type.getTypeStr(), name);
    auto &body = method->body().indent();
    auto value = "getExpr(" + std::to_string(dag.expressions[I].argNum) + ")";
    if (!type.isMLLExprClass()) {
      value = "llvm::cast<" + type.getTypeStrWithoutPtr() + ">(" + value + ")";
    }
    body << "return " << value << ";";
  }

  for (unsigned I = 0; I < dag.blocks.size(); ++I) {
    auto type = dag.blocks[I].type;
    auto name = dag.blocks[I].name;
    auto method =
        exprClass.addInlineMethod<Method::Const>(type.getTypeStr(), name);
    auto &body = method->body().indent();
    body << "return getBlock(" << dag.blocks[I].argNum << ");";
  }
}

static std::string getExprListForMethodExpr(Expr &expr) {
  ArgumentDAG dag = expr.getMethodArguments();
  std::string val = "ArrayRef<::mll::Expr*>{";
  bool hasOneValue = false;
  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto name = dag.getArgName(I);
    if (dag.isTypeConstraint(I)) {
      val += name.str() + ", ";
      hasOneValue = true;
    }
  }
  if (hasOneValue) {
    val.pop_back();
    val.pop_back();
  }
  val += "}";
  return val;
}

static std::string getExprList(Expr &expr) {

  if (expr.isBuiltinMethod()) {
    return getExprListForMethodExpr(expr);
  }
  ArgumentDAG dag = expr.getArguments();
  std::string val = "ArrayRef<::mll::Expr*>{";
  bool hasOneValue = false;
  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto type = dag.getArgType(I);
    auto fieldName = dag.getArgFieldName(I);
    auto name = dag.getArgName(I);
    if (type.isMLLExpr()) {
      val += name.str() + ", ";
      hasOneValue = true;
    }
  }
  if (hasOneValue) {
    val.pop_back();
    val.pop_back();
  }
  val += "}";
  return val;
}

static std::string getBlockList(Expr &expr) {
  ArgumentDAG dag = expr.getArguments();
  std::string val = "ArrayRef<::mll::Block*>{";
  bool hasOneValue = false;
  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto type = dag.getArgType(I);
    auto fieldName = dag.getArgFieldName(I);
    auto name = dag.getArgName(I);
    if (type.isBlock()) {
      val += name.str() + ", ";
      hasOneValue = true;
    }
  }
  if (hasOneValue) {
    val.pop_back();
    val.pop_back();
  }
  val += "}";
  return val;
}

static void defineConstructor(Class &exprClass, Expr &expr) {
  SmallVector<mlir::tblgen::MethodParameter> paramList;
  paramList.emplace_back("::mll::Location", "loc");
  addMethodParameters(expr, paramList);

  // TODO: Add assert for creating method expression types.
  auto *constructor = exprClass.addConstructor<Method::Private>(paramList);

  // Add member initalizer for constructor.
  std::string initString;
  llvm::raw_string_ostream initStr(initString);
  initStr << "loc, " << expr.getTypeIDExpr();
  initStr << ", std::make_pair(\"" << expr.getDialect().getName() << "\", \""
          << expr.getName() << "\")";
  initStr << ", type, " << getExprList(expr) << ", " << getBlockList(expr)
          << ", " << (expr.getExprKind());

  initStr.flush();
  constructor->addMemberInitializer("::mll::Expr", initString);

  if (expr.isBuiltinMethod()) {
    ArgumentDAG dag = expr.getMethodArguments();
    for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
      auto fieldName = dag.getArgFieldName(I);
      auto name = dag.getArgName(I);
      if (dag.isTypeConstraint(I)) {
        continue;
      }
      constructor->addMemberInitializer(fieldName, name);
    }
  } else {
    ArgumentDAG dag = expr.getArguments();
    for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
      if (dag.astNodes[I]) {
        continue;
      }
      auto fieldName = dag.getArgFieldName(I);
      auto name = dag.getArgName(I);
      constructor->addMemberInitializer(fieldName, name);
    }
  }

  ExtraClassDeclaration decl("friend class ::mll::MLLContext;\n");
  ExtraClassDeclaration decl2("friend class ::mll::ASTBuilder;\n");
  exprClass.declare<ExtraClassDeclaration>(decl);
  exprClass.declare<ExtraClassDeclaration>(decl2);
}

/// 2 = Argument number
static constexpr const char *methodExpr = R"(

auto &typeExprInfo{0} = typeExprList[{0}];
if (typeExprInfo{0}.isType) {{
  ::mll::emitError(loc, "Expected expression as argument {0}. But got type");
  return nullptr;
}

mll::Expr* {1} = typeExprInfo{0}.value.expr;
)";

/// 0 = variable name
/// 1 = Expected type
/// 2 = Argument number
static constexpr const char *methodExprTypeArg = R"(


auto &typeExprInfo{2} = typeExprList[{2}];
if (!typeExprInfo{2}.isType) {{
  ::mll::emitError(loc, "Expected Type as argument {2}. But got expression");
  return nullptr;
}

auto {0} = ::llvm::dyn_cast<{1}>(typeExprInfo{2}.value.type);
if (!{0}) {{
  ::mll::emitError(loc, "Invalid expression type for argument {2}. Expected type {1}");
  return nullptr;
}
)";

static void defineMethodEprInferTypeFunc(Class &exprClass, Expr &expr) {
  SmallVector<MethodParameter> paramList;
  paramList.push_back(MethodParameter("mll::ASTBuilder&", "builder"));
  paramList.push_back(MethodParameter("Location", "loc"));
  auto dag = expr.getMethodArguments();

  FmtContext ctx = FmtContext().withBuilder("builder");

  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto argName = dag.getArgName(I);
    ctx.addSubst(argName, argName);
    if (dag.isTypeConstraint(I)) {
      paramList.push_back(MethodParameter("mll::Expr* ", argName));
      continue;
    }

    auto argType = dag.getArgType(I);
    if (argType.isTypeArgument()) {
      paramList.push_back(MethodParameter(argType.getTypeStr(), argName));
      continue;
    }

    llvm::PrintFatalError(expr.getLoc(),
                          "MethodExpr can only Type arguments and expressions");
  }

  if (expr.getInferReturnTypeBody() != "") {
    auto method = exprClass.addStaticMethod("::mll::Type* ", "inferReturnType",
                                            paramList);
    auto &body = method->body().indent();
    body << tgfmt(expr.getInferReturnTypeBody(), &ctx).str() << "\n";
  }
  exprClass.addStaticMethod<Method::Declaration>("::mll::Type* ",
                                                 "inferReturnType", paramList);
}

static void defineInferTypeMethod(Class &exprClass, Expr &expr,
                                  bool isBuiltinMethod) {

  if (isBuiltinMethod) {
    defineMethodEprInferTypeFunc(exprClass, expr);
    return;
  }
  SmallVector<MethodParameter> paramList;
  paramList.push_back(MethodParameter("mll::ASTBuilder&", "builder"));
  paramList.push_back(MethodParameter("Location", "loc"));

  if (expr.getInferReturnTypeBody() != "") {
    auto method = exprClass.addStaticMethod("::mll::Type* ", "inferReturnType",
                                            paramList);
    auto &body = method->body().indent();
    FmtContext ctx = FmtContext().withBuilder("builder");
    body << tgfmt(expr.getInferReturnTypeBody(), &ctx).str() << "\n";
    return;
  }
  // FIXME: Any other way to declare a method?
  exprClass.addStaticMethod<Method::Declaration>("::mll::Type* ",
                                                 "inferReturnType", paramList);
}

static void defineMethodExprBuildFunc(Class &exprClass, Expr &expr) {
  SmallVector<MethodParameter> paramList;
  paramList.push_back(MethodParameter("mll::ASTBuilder&", "builder"));
  paramList.push_back(MethodParameter("Location", "loc"));
  auto dag = expr.getMethodArguments();
  paramList.push_back(MethodParameter(
      "::llvm::SmallVectorImpl<::mll::TypeExprInfo>& ", "typeExprList"));

  auto method = exprClass.addStaticMethod("::mll::Expr* ", "build", paramList);
  auto &body = method->body().indent();
  FmtContext ctx =
      FmtContext().withBuilder("builder").addSubst("_returnType", "returnType");

  std::string varNameStr;
  llvm::raw_string_ostream stream(varNameStr);

  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto argName = dag.getArgName(I);
    ctx.addSubst(argName, argName);
    stream << ", " << argName;

    if (dag.isTypeConstraint(I)) {
      body << llvm::formatv(methodExpr, I, argName);
      continue;
    }

    auto argType = dag.getArgType(I);

    if (argType.isTypeArgument()) {
      body << llvm::formatv(methodExprTypeArg, argName,
                            argType.getTypeStrWithoutPtr(), I);
      continue;
    }

    llvm::PrintFatalError(expr.getLoc(),
                          "MethodExpr can only Type arguments and expressions");
  }

  stream.flush();
  body << "auto returnType = " << expr.getFullClassName()
       << "::inferReturnType(builder, loc" << stream.str() << ");\n";
  // body << tgfmt(expr.getInferReturnTypeCode(), &ctx).str() << "\n";

  // FIXME: Do not allow builder to create builtin expressions.
  // Make build function as friend

  body << "return builder.create<" << expr.getFullClassName()
       << ">(loc, returnType" << stream.str() << ");";
}

static void defineBuildFunc(Class &exprClass, Expr &expr,
                            bool isBuiltinMethod) {

  if (isBuiltinMethod) {
    defineMethodExprBuildFunc(exprClass, expr);
    return;
  }
  SmallVector<MethodParameter> paramList;
  paramList.push_back(MethodParameter("mll::ASTBuilder&", "builder"));
  paramList.push_back(MethodParameter("Location", "loc"));
  auto method = exprClass.addStaticMethod("::mll::Expr* ", "build", paramList);
  auto &body = method->body().indent();
  body << "auto returnType = llvm::cast<"
       << expr.getReturnType().getTypeStrWithoutPtr() << ">("
       << expr.getFullClassName() << "::inferReturnType(builder,loc));\n";

  body << "return builder.create<" << expr.getFullClassName()
       << ">(loc, returnType);";
}

static void defineNameMethod(Class &exprClass, Expr &expr) {
  auto method = exprClass.addStaticInlineMethod("std::string", "getName");
  method->body().indent() << "return \"" << expr.getName() << "\";";

  // getUniqueName()
  method = exprClass.addStaticInlineMethod("std::string", "getUniqueName");
  method->body().indent() << "return \"" << expr.getUniqueName() << "\";";
}

static void defineStaticFunctions(Class &exprClass, Expr &expr) {
  auto method = exprClass.addStaticInlineMethod(
      "bool", "classof", MethodParameter("const ::mll::Expr* ", "base"));
  auto dummy = " return base->getTypeID() == ";
  method->body().indent() << dummy
                          << ::getTypeIDForClass(expr.getFullClassName())
                          << ";";

  auto method2 = exprClass.addStaticInlineMethod("enum ::mll::Expr::ExprKind",
                                                 "getExprKind");
  method2->body().indent() << "return " << expr.getExprKind() << ";";

  defineNameMethod(exprClass, expr);

  if (expr.isConstantExpr()) {
    auto method = exprClass.addStaticInlineMethod("std::string", "getTypeName");
    method->body().indent()
        << "return \"" << expr.getConstantType().getName() << "\";";
  }

  if (expr.isBuiltinMethodOrProp()) {
    defineBuildFunc(exprClass, expr, expr.isBuiltinMethod());
    defineInferTypeMethod(exprClass, expr, expr.isBuiltinMethod());
  }
}

static const char *const printStringFmt = R"(
mlir::raw_indented_ostream ros(os);
auto &fos = ros.indent();
os << "{0}: " << " {\n";
fos << "type: " << *getExprType() << "\n"; 
{1}
fos.unindent();
os << "}";
)";

static void defineASTPrintFunction(Class &exprClass, Expr &expr) {
  auto method = exprClass.addConstMethod(
      "void", "dump", MethodParameter("::llvm::raw_ostream &", "os"));
  method->markOverride();
  auto &body = method->body().indent();

  std::string printStr;
  ArgumentDAG dag =
      expr.isBuiltinMethod() ? expr.getMethodArguments() : expr.getArguments();
  llvm::raw_string_ostream print(printStr);
  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto fieldName = dag.getArgFieldName(I);
    std::string value = fieldName;
    if (expr.isBuiltinMethod()) {
      value = dag.getArgName(I).str() + "()";
    }
    if (dag.isTypeConstraint(I)) {
      value = "*" + value;
    } else {
      auto isPtrType = dag.getArgType(I).isASTNodeRelated();
      // De-reference the pointer.
      if (isPtrType) {
        value = "*" + value;
      }
    }
    print << "fos << \"" << dag.getArgName(I) << ": \" << " << value
          << " << \"\\n\";\n";
  }
  print.flush();
  auto exprName =
      expr.getDialect().getName().str() + "." + expr.getCppClassName().str();
  body << llvm::formatv(printStringFmt, exprName, printStr);
}

static void emitExprClass(Expr &expr, raw_ostream &os, bool isDef = false) {

  auto className = expr.getCppClassName();
  auto dialectName = expr.getDialect().getName();

  ::NamespaceEmitter emitter(os, {"mll", dialectName});

  mlir::tblgen::Class exprClass(className);
  exprClass.addParent(std::move(ParentClass("::mll::Expr")));

  declareFieldsAndMembers(exprClass, expr);

  defineConstructor(exprClass, expr);

  // Add static functions
  defineStaticFunctions(exprClass, expr);

  llvm::SmallVector<MethodParameter> params = {
      MethodParameter("::mll::Parser &", "parser"),
      MethodParameter("::mll::ASTBuilder &", "builder")};

  if (expr.isConstantExpr()) {
    params.push_back(MethodParameter("::mll::Type *", "ty"));
  }

  // Add parser declaration
  if (expr.needParserMethod()) {
    exprClass.declareStaticMethod(expr.getFullClassName() + "* ", "parse",
                                  params);
  } else {
    // FIXME: Accept from user?
    auto method = exprClass.addStaticMethod(expr.getFullClassName() + "* ",
                                            "parse", params);
    method->body().indent() << "return nullptr;";
  }

  // Add dump method
  defineASTPrintFunction(exprClass, expr);

  exprClass.finalize();

  if (!isDef) {
    exprClass.writeDeclTo(os);
  } else {
    exprClass.writeDefTo(os);
  }
}

static bool emitExprDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  std::vector<Record *> defs = recordKeeper.getAllDerivedDefinitions("Expr");
  for (auto &def : defs) {
    Expr expr(def);
    if (expr.getDialect().getName() != selectedDialect) {
      continue;
    }
    emitExprClass(expr, os);
    os << "MLIR_DECLARE_EXPLICIT_TYPE_ID(" << expr.getFullClassName() << ")\n";
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

static bool emitExprDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  std::vector<Record *> defs = recordKeeper.getAllDerivedDefinitions("Expr");

  ::Dialect d = getSingleDialect(recordKeeper);
  std::vector<std::string> propertyClasses, methodClasses;

  std::set<std::string> constExprTypes, varExprTypes;
  llvm::SmallVector<Expr> constExprs, varExprs;

  for (auto &def : defs) {
    Expr expr(def);

    if (expr.getDialect().getName() != selectedDialect) {
      continue;
    }
    if (expr.isPropertyExpr()) {
      propertyClasses.push_back(expr.getFullClassName());
    } else if (expr.isBuiltinMethod()) {
      methodClasses.push_back(expr.getFullClassName());
    } else if (expr.isConstantExpr()) {
      auto className = expr.getConstantType().getFullClassName();
      if (constExprTypes.find(className) == constExprTypes.end()) {
        constExprTypes.insert(expr.getConstantType().getFullClassName());
        constExprs.push_back(expr);
      }
    } else if (expr.isVariableExpr()) {
      auto className = expr.getVariableType().getFullClassName();
      if (varExprTypes.find(className) == varExprTypes.end()) {
        varExprTypes.insert(expr.getVariableType().getFullClassName());
        varExprs.push_back(expr);
      }
    }

    // Emit the ExprID explicit specializations to have a single symbol
    os << "MLIR_DEFINE_EXPLICIT_TYPE_ID(" << expr.getFullClassName() << ")\n";

    emitExprClass(expr, os, true);
  }

  mlir::raw_indented_ostream fos(os);

  // Register property expressions.
  auto method =
      Method("void", d.getFullCppClassName() + "::registerPropertyExprs",
             Method::None);
  auto &body = method.body().indent();

  for (auto &prop : propertyClasses) {
    body << "registerPropertyExpr<" << prop << ">();\n";
  }
  method.writeDefTo(fos, "");

  // Register method expressions
  method = Method("void", d.getFullCppClassName() + "::registerMethodExprs",
                  Method::None);
  auto &body1 = method.body().indent();

  for (auto &prop : methodClasses) {
    body1 << "registerMethodExpr<" << prop << ">();\n";
  }

  method.writeDefTo(fos, "");

  // Register cosntant expressions
  method =
      Method("void", d.getFullCppClassName() + "::registerConstantExprTypes",
             Method::None);
  auto &body2 = method.body().indent();

  for (auto &expr : constExprs) {
    auto funcName =
        "parse" + expr.getConstantType().getCppClassName().str() + "ConstExpr";
    auto parseMethod = Method("::mll::Expr *", funcName, Method::Declaration,
                              MethodParameter("::mll::Parser &", "parser"),
                              MethodParameter("::mll::ASTBuilder &", "builder"),
                              MethodParameter("::mll::Type *", "ty"));
    {
      auto dialectName = expr.getDialect().getName();

      ::NamespaceEmitter emitter(fos, {"mll", dialectName});
      parseMethod.writeDeclTo(fos);
    }
    body2 << "registerConstantExprTypeParsing<"
          << expr.getConstantType().getFullClassName()
          << ">((::mll::Dialect::MethodPtr)" << funcName << ");\n";
  }

  method.writeDefTo(fos, "");

  // Register variable expressions
  method =
      Method("void", d.getFullCppClassName() + "::registerVariableExprTypes",
             Method::None);
  auto &body3 = method.body().indent();

  for (auto &expr : varExprs) {
    auto funcName = "parse" + expr.getVariableType().getCppClassName().str() +
                    "VariableExpr";
    auto parseMethod = Method("::mll::Expr *", funcName, Method::Declaration,
                              MethodParameter("::mll::Parser &", "parser"),
                              MethodParameter("::mll::ASTBuilder &", "builder"),
                              MethodParameter("::mll::Type *", "ty"));
    {
      auto dialectName = expr.getDialect().getName();

      ::NamespaceEmitter emitter(fos, {"mll", dialectName});
      parseMethod.writeDeclTo(fos);
    }
    body3 << "registerVariableExprTypeParsing<"
          << expr.getVariableType().getFullClassName()
          << ">((::mll::Dialect::MethodPtr)" << funcName << ");\n";
  }

  method.writeDefTo(fos, "");
  return false;
}

} // namespace ns_expr

namespace ns_stmt {
static void addMethodParameters(Stmt &stmt,
                                SmallVector<MethodParameter> &paramList) {
  // Add the remaining arguments.
  ArgumentDAG dag = stmt.getArguments();
  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto type = dag.getArgType(I);
    paramList.emplace_back(type.getTypeStr(), dag.getArgName(I));
  }
}

static void declareFieldsAndMembers(Class &stmtClass, Stmt &stmt) {
  // Add the remaining arguments as field and access functions.
  ArgumentDAG dag = stmt.getArguments();
  for (unsigned I = 0; I < dag.others.size(); ++I) {
    auto type = dag.others[I].type;
    auto fieldName = dag.getArgFieldName(dag.others[I].index);
    auto name = dag.others[I].name;

    stmtClass.addField(type.getTypeStr(), fieldName);

    auto method =
        stmtClass.addInlineMethod<Method::Const>(type.getTypeStr(), name);
    auto &body = method->body().indent();
    body << "return " << fieldName << ";";
  }

  for (unsigned I = 0; I < dag.expressions.size(); ++I) {
    auto type = dag.expressions[I].type;
    auto name = dag.expressions[I].name;
    auto method =
        stmtClass.addInlineMethod<Method::Const>(type.getTypeStr(), name);
    auto &body = method->body().indent();

    auto value = "getExpr(" + std::to_string(dag.expressions[I].argNum) + ")";
    if (!type.isMLLExprClass()) {
      value = "llvm::cast<" + type.getTypeStrWithoutPtr() + ">(" + value + ")";
    }
    body << "return " << value << ";";
  }

  for (unsigned I = 0; I < dag.blocks.size(); ++I) {
    auto type = dag.blocks[I].type;
    auto name = dag.blocks[I].name;
    auto method =
        stmtClass.addInlineMethod<Method::Const>(type.getTypeStr(), name);
    auto &body = method->body().indent();
    if (type.isBlockList()) {
      assert(dag.blocks.size() == 1 &&
             "Only one blocklist argument is allowed");
      body << "return blocks();";
      continue;
    }
    body << "return getBlock(" << dag.blocks[I].argNum << ");";
  }
}

static std::string getExprList(Stmt &stmt) {
  ArgumentDAG dag = stmt.getArguments();
  std::string val = "ArrayRef<::mll::Expr*>{";
  bool hasOneValue = false;
  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto type = dag.getArgType(I);
    auto fieldName = dag.getArgFieldName(I);
    auto name = dag.getArgName(I);
    if (type.isMLLExpr()) {
      val += name.str() + ", ";
      hasOneValue = true;
    }
  }
  if (hasOneValue) {
    val.pop_back();
    val.pop_back();
  }
  val += "}";
  return val;
}

static std::string getBlockList(Stmt &stmt) {
  ArgumentDAG dag = stmt.getArguments();
  if (dag.blocks.size() == 1 && dag.blocks.front().type.isBlockList()) {
    return dag.blocks.front().name.str();
  }

  std::string val = "ArrayRef<::mll::Block*>{";
  bool hasOneValue = false;
  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto type = dag.getArgType(I);
    auto fieldName = dag.getArgFieldName(I);
    auto name = dag.getArgName(I);
    if (type.isBlock()) {
      val += name.str() + ", ";
      hasOneValue = true;
    }
  }
  if (hasOneValue) {
    val.pop_back();
    val.pop_back();
  }
  val += "}";
  return val;
}

static void defineConstructor(Class &stmtClass, Stmt &stmt) {
  SmallVector<mlir::tblgen::MethodParameter> paramList;
  paramList.emplace_back("::mll::Location", "loc");
  addMethodParameters(stmt, paramList);
  auto *constructor = stmtClass.addConstructor<Method::Private>(paramList);

  // Add member initalizer for constructor.
  std::string initString;
  llvm::raw_string_ostream initStr(initString);
  initStr << "loc, " << stmt.getTypeIDExpr();
  initStr << ", std::make_pair(\"" << stmt.getDialect().getName() << "\", \""
          << stmt.getName() << "\")";
  initStr << ", " << getExprList(stmt) << ", " << getBlockList(stmt);

  initStr.flush();
  constructor->addMemberInitializer("::mll::Stmt", initString);
  ArgumentDAG dag = stmt.getArguments();
  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    if (dag.astNodes[I]) {
      continue;
    }
    auto fieldName = dag.getArgFieldName(I);
    auto name = dag.getArgName(I);
    constructor->addMemberInitializer(fieldName, name);
  }

  ExtraClassDeclaration decl("friend class ::mll::MLLContext;\n");
  ExtraClassDeclaration decl2("friend class ::mll::ASTBuilder;\n");
  stmtClass.declare<ExtraClassDeclaration>(decl);
  stmtClass.declare<ExtraClassDeclaration>(decl2);
}

static void defineStaticFunctions(Class &stmtClass, Stmt &stmt) {
  auto method = stmtClass.addStaticInlineMethod(
      "bool", "classof", MethodParameter("const ::mll::Stmt* ", "base"));
  auto dummy = " return base->getTypeID() == ";
  method->body().indent() << dummy << stmt.getTypeIDExpr() << ";";

  // getName()
  method = stmtClass.addStaticInlineMethod("std::string", "getName");
  method->body().indent() << "return \"" << stmt.getName() << "\";";

  // getUniqueName()
  method = stmtClass.addStaticInlineMethod("std::string", "getUniqueName");
  method->body().indent() << "return \"" << stmt.getUniqueName() << "\";";

  // hasCustomParseMethod()
  method = stmtClass.addStaticInlineMethod("bool", "hasCustomParseMethod");
  method->body().indent() << "return " << stmt.needParserMethod() << ";";
}

static const char *const printStringFmt = R"(
mlir::raw_indented_ostream ros(os);
auto &fos = ros.indent();
os << "{0}: " << " {\n";
{1}
fos.unindent();
os << "}\n";
)";
static void defineASTPrintFunction(Class &stmtClass, Stmt &stmt) {
  auto method = stmtClass.addConstMethod(
      "void", "dump", MethodParameter("::llvm::raw_ostream &", "os"));
  method->markOverride();
  auto &body = method->body().indent();

  std::string printStr;
  ArgumentDAG dag = stmt.getArguments();
  llvm::raw_string_ostream print(printStr);
  for (unsigned I = 0; I < dag.getNumArgs(); ++I) {
    auto fieldName = dag.getArgFieldName(I);

    std::string value = fieldName;
    auto isPtrType = dag.getArgType(I).isASTNodeRelated();
    // De-reference the pointer.
    if (isPtrType) {
      value = "*" + value;
    }
    print << "fos << \"" << dag.getArgName(I) << ": \" << " << value
          << " << \"\\n\";\n";
  }
  print.flush();
  auto stmtName =
      stmt.getDialect().getName().str() + "." + stmt.getCppClassName().str();
  body << llvm::formatv(printStringFmt, stmtName, printStr);
}

static void emitStmtClass(Stmt &stmt, raw_ostream &os, bool isDef = false) {

  auto className = stmt.getCppClassName();
  auto dialectName = stmt.getDialect().getName();

  ::NamespaceEmitter emitter(os, {"mll", dialectName});

  mlir::tblgen::Class stmtClass(className);
  stmtClass.addParent(std::move(ParentClass("::mll::Stmt")));

  declareFieldsAndMembers(stmtClass, stmt);

  defineConstructor(stmtClass, stmt);

  // Add static functions
  defineStaticFunctions(stmtClass, stmt);

  // Add parser declaration
  if (stmt.needParserMethod()) {
    stmtClass.declareStaticMethod(
        stmt.getFullClassName() + "* ", "parse",
        MethodParameter("::mll::Parser &", "parser"),
        MethodParameter("::mll::ASTBuilder &", "builder"));
  }

  // Add parser declaration
  if (stmt.needParserMethod()) {
    stmtClass.declareStaticMethod(
        stmt.getFullClassName() + "* ", "parse",
        MethodParameter("::mll::Parser &", "parser"),
        MethodParameter("::mll::ASTBuilder &", "builder"));
  } else {
    // FIXME: Accept from user?
    auto method = stmtClass.addStaticMethod(
        stmt.getFullClassName() + "* ", "parse",
        MethodParameter("::mll::Parser &", "parser"),
        MethodParameter("::mll::ASTBuilder &", "builder"));
    method->body().indent() << "return nullptr;";
  }

  // Add dump method
  defineASTPrintFunction(stmtClass, stmt);

  stmtClass.finalize();

  if (!isDef) {
    stmtClass.writeDeclTo(os);
  } else {
    stmtClass.writeDefTo(os);
  }
}

static bool emitStmtDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  std::vector<Record *> defs = recordKeeper.getAllDerivedDefinitions("Stmt");
  for (auto &def : defs) {
    Stmt stmt(def);
    if (stmt.getDialect().getName() != selectedDialect) {
      continue;
    }
    emitStmtClass(stmt, os);
    os << "MLIR_DECLARE_EXPLICIT_TYPE_ID(" << stmt.getFullClassName() << ")\n";
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

static bool emitStmtDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  std::vector<Record *> defs = recordKeeper.getAllDerivedDefinitions("Stmt");

  std::vector<std::string> stmtList;
  std::string fullDialectName =
      getSingleDialect(recordKeeper).getFullCppClassName();

  for (auto &def : defs) {
    Stmt stmt(def);

    if (stmt.getDialect().getName() != selectedDialect) {
      continue;
    }
    // Emit the StmtID explicit specializations to have a single symbol
    os << "MLIR_DEFINE_EXPLICIT_TYPE_ID(" << stmt.getFullClassName() << ")\n";

    emitStmtClass(stmt, os, true);

    stmtList.push_back(stmt.getFullClassName());
  }

  auto funcName = fullDialectName + "::registerStmts";
  auto method =
      mlir::tblgen::Method("void", funcName, Method::Properties::None);
  auto &body = method.body().indent();
  for (auto &type : stmtList) {
    body << "registerStmt<" << type << ">();\n";
  }

  mlir::raw_indented_ostream fos(os);
  method.writeDefTo(fos, "");

  return false;
}

} // namespace ns_stmt

static bool emitBinaryOperatorDefs(const RecordKeeper &recordKeeper,
                                   raw_ostream &os) {
  std::vector<Record *> defs =
      recordKeeper.getAllDerivedDefinitions("BinaryOperator");

  ::Dialect d = ns_stmt::getSingleDialect(recordKeeper);
  auto method =
      Method("void", d.getFullCppClassName() + "::registerBinaryOperators",
             Method::None);
  auto &body = method.body().indent();
  for (auto &def : defs) {
    BinaryOperator binaryOperator(def);
    if (binaryOperator.getDialect().getName() != selectedDialect) {
      continue;
    }
    body << "context->registerBinaryOperator(::mll::BinaryOperator(this, \""
         << binaryOperator.getKeyword() << "\", "
         << binaryOperator.isRelational() << ", "
         << binaryOperator.getPrecedence() << "));\n";
  }
  mlir::raw_indented_ostream fos(os);
  method.writeDefTo(fos, "");
  return false;
}

static bool emitNodeDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("ASTNode Declarations", os);
  ns_expr::emitExprDecls(recordKeeper, os);
  ns_stmt::emitStmtDecls(recordKeeper, os);
  return false;
}

static bool emitNodeDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("ASTNode Definitions", os);
  ns_expr::emitExprDefs(recordKeeper, os);
  ns_stmt::emitStmtDefs(recordKeeper, os);
  emitBinaryOperatorDefs(recordKeeper, os);
  return false;
}

static mlir::GenRegistration
    genNodeDecls("gen-ast-node-decls", "Generate AST Node declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   return emitNodeDecls(records, os);
                 });

static mlir::GenRegistration
    genNodeDefs("gen-ast-node-defs", "Generate AST Node definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  return emitNodeDefs(records, os);
                });