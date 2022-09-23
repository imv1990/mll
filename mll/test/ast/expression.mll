// RUN: %mll %s -emit-ast | FileCheck %s
// Simple assignment
a = 3 + 3

// CHECK: builtin.BinaryOpExpr