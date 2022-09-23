// RUN: %mll %s -emit-ast | FileCheck %s
a = builtin.i32_min
b = i32_min

// CHECK: builtin.I32MinPropertyExpr
// CHECK: builtin.I32MinPropertyExpr