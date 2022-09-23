// RUN: %mll %s | FileCheck %s
x = array<4*4*i32>.dense(0)

for i in range(4) {
  for j in range(4) {
    x[i, j] = i + j
  }
}

print(x)

//CHECK: 0,   1,   2,   3 
//CHECK: 1,   2,   3,   4 
//CHECK: 2,   3,   4,   5 
//CHECK: 3,   4,   5,   6