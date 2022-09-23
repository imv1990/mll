// RUN: %mll %s -emit-mlir | FileCheck %s
a = 3 + 5 * 4
b = 3.2
vinay = 1.2

// CHECK: func.func @main
// CHECK: memref.alloca
// CHECK: memref.alloca
// CHECK: memref.alloca
// CHECK: arith.constant 3
// CHECK: memref.store
// CHECK: arith.constant 3.200
// CHECK: memref.store