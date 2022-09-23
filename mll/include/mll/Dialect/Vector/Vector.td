include "mll/Parse/Builtin.td"

def Vector_Dialect : Dialect {
  let name = "vector";
}

def Vector_Type : Type<Vector_Dialect, "Vector", "vector"> {
  let parameters = (ins VecIntType:$shape, AnyCPPType:$base);

  let declareParserMethod = 1;
}

def VectorSplatConst_Expr : ConstantExpr<Vector_Type, "VectorSplat"> {
  let arguments = (ins AnyExpr:$value);
}

def Vector : TypeConstraint<"mll::vector::VectorType">;

def VectorTypeArg : TypeArg<"::mll::vector::VectorType *">;

def : MethodExpr<Vector_Dialect, "Load", "load"> {
  let methodArgs = (ins VectorTypeArg:$vectorTy, Array:$arr, List:$indices);

  let result = Vector;

  let inferReturnTypeBody = [{ return $vectorTy; }];
}

def : MethodExpr<Vector_Dialect, "ReduceAdd", "reduce_add"> {
  let methodArgs = (ins Vector:$arg);

  let result = AnyTy;

  let inferReturnTypeBody = [{ return llvm::cast<::mll::vector::VectorType>($arg->getExprType())->base(); }];
}