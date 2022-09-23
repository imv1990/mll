include "mll/Parse/Builtin.td"

def OMP_Dialect : Dialect {
  let name = "omp";
}

def Parallel_Stmt : Stmt<OMP_Dialect, "Parallel", "parallel"> {
  let arguments = (ins Block:$body, SymbolTable:$symTable);

}

def Critical_Stmt : Stmt<OMP_Dialect, "Critical", "critical"> {
  let arguments = (ins Block:$body, SymbolTable:$symTable);

}