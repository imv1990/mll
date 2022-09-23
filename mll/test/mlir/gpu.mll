// RUN: %mll %s -emit-mlir | FileCheck %s
a = array<10*i32>.dense(1)

gpu.host_register(a)

gpu.launch blocks=(1, 1, 1) threads=(10, 1, 1) {
 a[threadIdx] = 10
}


//CHECK: gpu.launch blocks
//CHECK: memref.store
//CHECK: gpu.terminator