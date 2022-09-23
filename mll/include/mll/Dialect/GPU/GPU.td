include "mll/Parse/Builtin.td"

def GPU_Dialect : Dialect {
  let name = "gpu";
}

// Register array arguments.
def Host_Register_Stmt : Stmt<GPU_Dialect, "HostRegister", "host_register"> {
  let arguments = (ins AnyListExpr:$args);
}

def Launch_Stmt : Stmt<GPU_Dialect, "Launch", "launch"> {
  let arguments = (ins AnyListExpr:$blocks, AnyListExpr:$threads, Block:$body, SymbolTable:$symTable);
}