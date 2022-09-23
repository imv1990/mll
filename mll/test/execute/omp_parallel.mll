// RUN: %mll %s | FileCheck %s
a = array<10*10*i32>.dense(1)

for i in range(10) {
  omp.parallel {
    for j in range(10) {
      a[i,j] =  100
    }
  }
}

print(a)

//CHECK: 100, 100
