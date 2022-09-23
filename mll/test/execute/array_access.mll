// RUN: %mll %s | FileCheck %s
x = array<3*4*f32>.dense(10.2)

print ("val = ", x[0, 1])

x[1, 2] = 5.2

print ("val again = ", x)

// CHECK: 10.2,   10.2,   10.2,   10.2
// CHECK: 10.2,   10.2,   5.2,   10.2 
// CHECK: 10.2,   10.2,   10.2,   10.2
