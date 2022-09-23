// RUN: %mll %s | FileCheck %s
a = array<10*10*i32>.dense(1)

for i in range(10) {
  omp.parallel {
    for j in range(10) {
      a[i,j] =  100
    }
    omp.critical {
      print("i = ",i)
    }
  }
}

print(a)

//CHECK: i = 
//CHECK: 100, 100