// RUN: %mll %s | FileCheck %s
a = vector.vector<8*i32>.splat(10)

print("vector = ", a)


// CHECK: vector =  ( 10, 10, 10, 10, 10, 10, 10, 10 )