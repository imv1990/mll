// RUN: %mll %s -emit-ast | FileCheck %s

a = 3 + 5 * 4
print ("a = ", a)

//CHECK: Symbol: a
//CHECK: builtin.PrintStmt

