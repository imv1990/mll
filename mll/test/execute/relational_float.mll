
// RUN: %mll %s | FileCheck %s

// CHECK: True
print(3.45 < 6.1)
// CHECK: True
print(3.45 <= 6.1)
// CHECK: False
print(3.45 > 6.1)
// CHECK: False
print(3.450 >= 6.1)
// CHECK: False
print(3.45 == 6.1)
// CHECK: True
print(3.45 != 6.1)
// CHECK: False
print(3.45 != 6.1 and 3.45 >  6.1)
// CHECK: True
print(3.45 != 6.1 or 3.45 == 6.1)

