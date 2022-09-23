// RUN: %mll %s | FileCheck %s

func funcName(a : i32, b : i32, arr : array<10*20*i32>) -> i32{
  print (a, b)
  return (a + b + arr[0,0])
}

arr = array<10*20*i32>.dense(10)
print(funcName(10, 20, arr))


// CHECK: 10 20
// CHECK: 40
