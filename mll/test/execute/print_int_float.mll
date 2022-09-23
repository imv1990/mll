// RUN: %mll %s | FileCheck %s
a = 10 + 12
b = 3.2 + 6.8
print("sum int = ", a)
print("sum float = ", b)


// CHECK: sum int = 22
// CHECK: sum float = 10.000000
