// RUN: %mll %s | FileCheck %s
a = 10
if (3 < 4 and 4 > 5) {
  a = 20
} else if ( 4 > 5) {
 a = 30
} else  {
 a = 45 + 10
}

print(a)

// CHECK: 55