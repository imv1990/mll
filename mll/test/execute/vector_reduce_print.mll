// RUN: %mll %s | FileCheck %s
a = array<80*i32>.dense(10)

b = vector.load(vector.vector<4*i32>, a, [0])

print(vector.reduce_add( b + b))

// CHECK: 80