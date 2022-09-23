// RUN: %mll %s -emit-ast | FileCheck %s
// Simple assignment
a = 3
b = 3.2

// CHECK: builtin.AssignStmt
// CHECK: Symbol: a
// CHECK: Type: builtin.I32Type
// CHECK: builtin.AssignStmt
// CHECK: Type: builtin.F32Type