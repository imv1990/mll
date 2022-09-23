
// RUN: %mll %s | FileCheck %s
a = array<3*3*i32>.dense(10)
b = array<3*3*i32>.dense(20)

c = a + b

print("sum = ", c)


//CHECK: 30,   30,   30 
//CHECK: 30,   30,   30 
//CHECK: 30,   30,   30