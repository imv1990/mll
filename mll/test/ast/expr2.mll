// RUN: %mll %s -emit-ast | FileCheck %s
// Simple assignment
a = (3 + 3 - 7 / 10) * (4)

// CHECK: builtin.BinaryOpExpr
// CHECK op: +
// CHECK op: *
// CHECK op: -
// CHECK op: /