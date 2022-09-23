// RUN: %mll %s -emit-ast | FileCheck %s
// Simple assignment
a = 3

// CHECK: builtin.AssignStmt
// CHECK: Symbol: a
// CHECK: Type: builtin.I32Type