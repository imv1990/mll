include "mll/AST/ASTBase.td"

def Builtin_Dialect : Dialect {
  let name = "builtin";
}

// ----------------------- Defining Types --------//
def I32_Type : Type<Builtin_Dialect, "I32", "i32">;
def F32_Type : Type<Builtin_Dialect, "F32", "f32">;

def I1_Type : Type<Builtin_Dialect, "I1", "i1">;

def None_Type: Type<Builtin_Dialect, "None", "none">;

def String_Type : Type<Builtin_Dialect , "String", "string">;

def List_Type : Type<Builtin_Dialect, "List", "list">;

def Array_Type : Type<Builtin_Dialect, "Array", "array"> {
  let parameters = (ins VecIntType:$shape, AnyCPPType:$base);

  let declareParserMethod = 1;
}

def Function_Type : Type<Builtin_Dialect, "Function", "function"> {
  let parameters = (ins AnyCPPType:$returnType, CPPTypeList:$args);
}

// ----------------------- Defining Expressions --------//

// Expressions
def Symbol_Expr : Expr<Builtin_Dialect, "Symbol", "symbol_expr"> {
  let arguments = (ins Symbol:$symbol);

  let declareParserMethod = 0;
}

// Constant expressions for types.
class Constant_Expr<Type T, CPPType C> : ConstantExpr<T, T.cppName> {
  let arguments = (ins C:$value);
}

def : Constant_Expr<I32_Type, Integer>;
def : Constant_Expr<F32_Type, Float>;
def : Constant_Expr<String_Type, AnyStr>;
def : Constant_Expr<None_Type, Integer>;

def ArrayConst_Expr : ConstantExpr<Array_Type, "Array"> {
  let arguments = (ins AnyExpr:$value);
}

// Represents array<>.dense() expression.
def DenseArrayConst_Expr : ConstantExpr<Array_Type, "DenseArray"> {
  let arguments = (ins AnyExpr:$value);
}

def BinaryOp_Expr : Expr<Builtin_Dialect, "BinaryOp", "binary_op"> {
  let arguments = (ins AnyExpr:$lhs, AnyExpr:$rhs, AnyStr:$op);

  let declareParserMethod = 0;
}

def Call_Expr : Expr<Builtin_Dialect, "Call", "call"> {
  let arguments = (ins Symbol:$func, AnyListExpr:$args);
}

/// Variable expressions
def : VariableExpr<Array_Type, "ArrayAccess"> {
  let arguments = (ins Symbol:$base, AnyListExpr:$indices);
}

// Types to be used.
def AnyTy : TypeConstraint<"mll::builtin::Type">;
def I32 : TypeConstraint<"mll::builtin::I32Type">;
def F32 : TypeConstraint<"mll::builtin::F32Type">;
def Array : TypeConstraint<"mll::builtin::ArrayType">;
def List : TypeConstraint<"mll::builtin::ListType">;

def FunctionType : MLLCPPType<"::mll::builtin::FunctionType *">;

def I32TypeArg : TypeArg<"::mll::builtin::I32Type*">;
def F32TypeArg : TypeArg<"::mll::builtin::F32Type*">;

def : PropertyExpr<Builtin_Dialect, "I32Min", "i32_min", I32_Type> {
  let inferReturnTypeBody = [{ return $_builder.getType<I32Type>(); }];
}

def : MethodExpr<Builtin_Dialect, "AddI", "addi"> {
  let methodArgs = (ins I32:$lhs, I32:$rhs);
  let result = I32;
  let inferReturnTypeBody = [{ return $lhs->getExprType(); }];
}

// Binary operators:
def : BinaryOperator<Builtin_Dialect, "+", 10>;
def : BinaryOperator<Builtin_Dialect, "-", 10>;
def : BinaryOperator<Builtin_Dialect, "*", 20>;
def : BinaryOperator<Builtin_Dialect, "/", 20>;
def : BinaryOperator<Builtin_Dialect, "<", 8, 1>;
def : BinaryOperator<Builtin_Dialect, ">", 8, 1>;
def : BinaryOperator<Builtin_Dialect, "<=", 8, 1>;
def : BinaryOperator<Builtin_Dialect, ">=", 8, 1>;
def : BinaryOperator<Builtin_Dialect, "==", 8, 1>;
def : BinaryOperator<Builtin_Dialect, "!=", 8, 1>;
def : BinaryOperator<Builtin_Dialect, "and", 5, 1>;
def : BinaryOperator<Builtin_Dialect, "or", 5, 1>;

// Statements

// Generic statements to carry expressions with none return type and
// added as new statements.
// Like func_call(), memref.store, etc
def Expr_Stmt : Stmt<Builtin_Dialect, "Expr", "expr_stmt"> {
  let arguments = (ins AnyExpr:$arg);

  let declareParserMethod = 0;
}

def Assign_Stmt : Stmt<Builtin_Dialect, "Assign", "assign"> {
  let arguments = (ins AnyExpr:$lhs, AnyExpr:$rhs);

  let declareParserMethod = 0;
}

def Print_Stmt : Stmt<Builtin_Dialect, "Print", "print"> {
  let arguments = (ins AnyListExpr:$args);
}

def For_Stmt : Stmt<Builtin_Dialect, "For", "for"> {
  let arguments = (ins Symbol:$iv, AnyExpr:$lb, AnyExpr:$ub, AnyExpr:$step, SymbolTable:$symTable, Block:$body);
}

def If_Stmt : Stmt<Builtin_Dialect, "If", "if"> {
  let arguments = (ins AnyListExpr:$conditions, BlockList:$blocksList, SymTableList:$symTables);
}

def Func_Stmt : Stmt<Builtin_Dialect, "Func", "func"> {
  let arguments = (ins AnyStr:$name, SymbolList:$args, SymbolTable:$symTable, FunctionType:$funcType, Block:$body);
}

def Return_Stmt : Stmt<Builtin_Dialect, "Return", "return"> {
  let arguments = (ins AnyExpr:$arg);
}


class UnaryArrayExpr<string className, string name> : MethodExpr<Builtin_Dialect, className, name> {
  let methodArgs = (ins Array:$arg);
  let result = Array;
  let inferReturnTypeBody = [{ return $arg->getExprType(); }];
}

def : UnaryArrayExpr<"Exp", "exp">;
def : UnaryArrayExpr<"Sum", "sum">;
def : UnaryArrayExpr<"Max", "max">;
