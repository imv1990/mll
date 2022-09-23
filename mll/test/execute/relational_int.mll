
// RUN: %mll %s | FileCheck %s

// CHECK: True
print(3 < 6)
// CHECK: True
print(3 <= 6)
// CHECK: False
print(3 > 6)
// CHECK: True
print(30 >= 6)
// CHECK: False
print(3 == 6)
// CHECK: True
print(3 != 6)
// CHECK: False
print(3 != 6 and 3 >  6)
// CHECK: True
print(3 != 6 or 3 == 6)

