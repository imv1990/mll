
// Represents the base Dialect class.
class Dialect {
  string name = ?;
}

class Constraint<string c> {
  string cond = c;

  string isa = "::llvm::isa<" # c # ">";
}

class TypeConstraint<string c>  : Constraint<c> {
}

// Represents any C++ class types in string.
// classes pointers, etc.
class CPPType<string str> {
  string c = str;
}

class MLLCPPType<string str> : CPPType<str>  {

}

// Base class for all expression related classes.
class MLLExprType<string str> : CPPType<str> {

}

class VectorCPPType<CPPType ty> : CPPType<"llvm::SmallVector<" # ty.c # ">"> {
  CPPType baseTy = ty;
}

// Different nodes used in building expressions and statements.
def Symbol : CPPType<"::mll::Symbol*">;

def SymbolList : VectorCPPType<Symbol>;

def SymbolTable : CPPType<"::mll::SymbolTable*">;

def SymTableList : VectorCPPType<SymbolTable>;

def AnyInt  : CPPType<"::llvm::APInt">;

def Integer : CPPType<"int">;

def Int64 : CPPType<"int64_t">;

def VecIntType : VectorCPPType<Int64>;

def Float : CPPType<"float">;

def Block : CPPType<"::mll::Block*">;

def AnyExpr : MLLExprType<"::mll::Expr*">;

/// FIXME: Currently used as a way to get variadic arguments
/// Should be changed to just vector of expressions.
def AnyListExpr : MLLExprType<"::mll::builtin::ListExpr*">;

def BlockList : VectorCPPType<Block>;

def AnyStmt : MLLExprType<"::mll::Stmt*">;

def AnyCPPType : MLLCPPType<"::mll::Type*">;

def CPPTypeList : VectorCPPType<AnyCPPType>;

def AnyStr : CPPType<"std::string">;

// Operator for parameters


def ins;

// Operator for outputs
def outs;

// Base class for defining types.
// TODO : Add parser / printer/ MLIR conversion methods.
class Type<Dialect d, string cppNameStr, string typeName> {

  Dialect dialect = d;

  string cppName = cppNameStr;

  string cppClassName = cppNameStr # "Type";

  string name = typeName;

  dag parameters = (ins);

  bit declareParserMethod = 0;

  TypeConstraint predicate = TypeConstraint<"llvm::isa<::mll::" # dialect.name #
                             "::" # cppClassName # ">()">;

  MLLCPPType cppType = MLLCPPType<"::mll::" # dialect.name # "::" # cppClassName # "*">;
}

// Base class for defining Generic expressions.
class Expr<Dialect d, string cppName, string nameVal> {
  Dialect dialect = d;

  string cppClassName = cppName # "Expr";

  dag arguments = (ins);

  string name = nameVal;

  MLLCPPType exprType = AnyCPPType;

  bit declareParserMethod = 1;

  Constraint predicate = Constraint<"llvm::isa<::mll::" # dialect.name #
                             "::" # cppClassName # ">()">;

  code extraClassDeclaration = "";

  MLLExprType cppType = MLLExprType<"::mll::" # dialect.name # "::" # cppClassName # "*">;
}

// Class to build the constant expressions for the given type
class ConstantExpr<Type type, string name> : Expr<type.dialect, name # "Constant", name # "_" # type.name # "_constant"> {

  // Type of the constant
  Type constantType = type;

  // expression type.
  let exprType = type.cppType;

  // No need to parse it. Do not override.
  let declareParserMethod = 0;
}

class VariableExpr<Type type, string name>: Expr<type.dialect, name # "Variable", name # "_" # type.name # "_variable"> {
  Type variableType = type;

  let exprType = AnyCPPType;

  // No need to parse it. Do not override.
  let declareParserMethod  = 0; 
}

// Expression like builtin function call.
// Syntax is <dialect-name>.name`(`arguments`)`
// Arguments can only be of expression type.
class PropertyExpr<Dialect d, string className, string builtinName, Type type> : Expr<d, className # "Property", builtinName> {

  string name = builtinName;

  let exprType = type.cppType;

  let declareParserMethod = 0;

  code inferReturnTypeBody = ?;
}

// Expression like builtin function call.
// Syntax is <dialect-name>.name`(`arguments`)`
// Arguments can only be of expression type.
class MethodExpr<Dialect d, string className, string builtinName> : Expr<d, className # "Method", builtinName> {

  string name = builtinName;

  let exprType = AnyCPPType;

  dag methodArgs= (ins);

  TypeConstraint result = ?;

  let declareParserMethod = 0;

  code inferReturnTypeBody = "";
}

// Used when the MethodExpr wants to take type as input in the language.
// Example: builtin.splat(vector<64xi32>, b). Here first argument
// is mentioned as TypeArg in methodArgs.
class TypeArg<string c> : CPPType<c> {
}
def AnyTypeArg : TypeArg<"::mll::Type*">;

// Base class for defining Generic Statements.
class Stmt<Dialect d, string cppName, string nameVal> {
  Dialect dialect = d;

  // unique identifier for statement.
  string name = nameVal;

  string cppClassName = cppName # "Stmt";

  dag arguments = (ins);

  bit declareParserMethod = 1;

  Constraint predicate = Constraint<"llvm::isa<::mll::" # dialect.name #
                             "::" # cppClassName # ">()">;

  code extraClassDeclaration = "";

  MLLExprType cppType = MLLExprType<"::mll::" # dialect.name # "::" # cppClassName # "*">;
}

// Class used to register the binary operator for the given dialect.
// valid characters in operators are
// + | - | / | * | [a-zA-Z]+
class BinaryOperator<Dialect d, string keyword, int precedenceVal, bit isRelational = 0> {
  Dialect dialect = d;
  string kw = keyword;
  int precedence = precedenceVal;
  bit isRelationalKind = isRelational;
}

