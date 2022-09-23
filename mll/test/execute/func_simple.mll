// RUN: %mll %s | FileCheck %s

func funcName(a : i32, b : i32) -> i32{
  print (a, b)
  return (a + b)
}

print(funcName(10, 20))


// CHECK: 10 20
// CHECK: 30