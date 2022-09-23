
// RUN: %mll %s -emit-ast | FileCheck %s

a = builtin.addi(10, 11) + 3
b = addi(a, 11)

// CHECK: builtin.AddIMethodExpr
// CHECK: builtin.AddIMethodExpr