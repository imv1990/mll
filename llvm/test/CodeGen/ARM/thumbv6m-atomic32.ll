; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=thumbv6m-none-eabi < %s | FileCheck %s --check-prefixes=CHECK,NO-ATOMIC32
; RUN: llc -mtriple=thumbv6m-none-eabi -mattr=+atomics-32 < %s | FileCheck %s --check-prefixes=CHECK,ATOMIC32

define i8 @load8(ptr %p) {
; NO-ATOMIC32-LABEL: load8:
; NO-ATOMIC32:       @ %bb.0:
; NO-ATOMIC32-NEXT:    .save {r7, lr}
; NO-ATOMIC32-NEXT:    push {r7, lr}
; NO-ATOMIC32-NEXT:    movs r1, #5
; NO-ATOMIC32-NEXT:    bl __atomic_load_1
; NO-ATOMIC32-NEXT:    pop {r7, pc}
;
; ATOMIC32-LABEL: load8:
; ATOMIC32:       @ %bb.0:
; ATOMIC32-NEXT:    ldrb r0, [r0]
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    bx lr
  %v = load atomic i8, ptr %p seq_cst, align 1
  ret i8 %v
}

define void @store8(ptr %p) {
; NO-ATOMIC32-LABEL: store8:
; NO-ATOMIC32:       @ %bb.0:
; NO-ATOMIC32-NEXT:    .save {r7, lr}
; NO-ATOMIC32-NEXT:    push {r7, lr}
; NO-ATOMIC32-NEXT:    movs r1, #0
; NO-ATOMIC32-NEXT:    movs r2, #5
; NO-ATOMIC32-NEXT:    bl __atomic_store_1
; NO-ATOMIC32-NEXT:    pop {r7, pc}
;
; ATOMIC32-LABEL: store8:
; ATOMIC32:       @ %bb.0:
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    movs r1, #0
; ATOMIC32-NEXT:    strb r1, [r0]
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    bx lr
  store atomic i8 0, ptr %p seq_cst, align 1
  ret void
}

define i8 @rmw8(ptr %p) {
; NO-ATOMIC32-LABEL: rmw8:
; NO-ATOMIC32:       @ %bb.0:
; NO-ATOMIC32-NEXT:    .save {r7, lr}
; NO-ATOMIC32-NEXT:    push {r7, lr}
; NO-ATOMIC32-NEXT:    movs r1, #1
; NO-ATOMIC32-NEXT:    movs r2, #5
; NO-ATOMIC32-NEXT:    bl __atomic_fetch_add_1
; NO-ATOMIC32-NEXT:    pop {r7, pc}
;
; ATOMIC32-LABEL: rmw8:
; ATOMIC32:       @ %bb.0:
; ATOMIC32-NEXT:    .save {r7, lr}
; ATOMIC32-NEXT:    push {r7, lr}
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    movs r1, #1
; ATOMIC32-NEXT:    bl __sync_fetch_and_add_1
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    pop {r7, pc}
  %v = atomicrmw add ptr %p, i8 1 seq_cst, align 1
  ret i8 %v
}

define i8 @cmpxchg8(ptr %p) {
; NO-ATOMIC32-LABEL: cmpxchg8:
; NO-ATOMIC32:       @ %bb.0:
; NO-ATOMIC32-NEXT:    .save {r7, lr}
; NO-ATOMIC32-NEXT:    push {r7, lr}
; NO-ATOMIC32-NEXT:    .pad #8
; NO-ATOMIC32-NEXT:    sub sp, #8
; NO-ATOMIC32-NEXT:    add r1, sp, #4
; NO-ATOMIC32-NEXT:    movs r2, #0
; NO-ATOMIC32-NEXT:    strb r2, [r1]
; NO-ATOMIC32-NEXT:    movs r3, #5
; NO-ATOMIC32-NEXT:    str r3, [sp]
; NO-ATOMIC32-NEXT:    movs r2, #1
; NO-ATOMIC32-NEXT:    bl __atomic_compare_exchange_1
; NO-ATOMIC32-NEXT:    ldr r0, [sp, #4]
; NO-ATOMIC32-NEXT:    add sp, #8
; NO-ATOMIC32-NEXT:    pop {r7, pc}
;
; ATOMIC32-LABEL: cmpxchg8:
; ATOMIC32:       @ %bb.0:
; ATOMIC32-NEXT:    .save {r7, lr}
; ATOMIC32-NEXT:    push {r7, lr}
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    movs r1, #0
; ATOMIC32-NEXT:    movs r2, #1
; ATOMIC32-NEXT:    bl __sync_val_compare_and_swap_1
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    pop {r7, pc}
  %res = cmpxchg ptr %p, i8 0, i8 1 seq_cst seq_cst
  %res.0 = extractvalue { i8, i1 } %res, 0
  ret i8 %res.0
}

define i16 @load16(ptr %p) {
; NO-ATOMIC32-LABEL: load16:
; NO-ATOMIC32:       @ %bb.0:
; NO-ATOMIC32-NEXT:    .save {r7, lr}
; NO-ATOMIC32-NEXT:    push {r7, lr}
; NO-ATOMIC32-NEXT:    movs r1, #5
; NO-ATOMIC32-NEXT:    bl __atomic_load_2
; NO-ATOMIC32-NEXT:    pop {r7, pc}
;
; ATOMIC32-LABEL: load16:
; ATOMIC32:       @ %bb.0:
; ATOMIC32-NEXT:    ldrh r0, [r0]
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    bx lr
  %v = load atomic i16, ptr %p seq_cst, align 2
  ret i16 %v
}

define void @store16(ptr %p) {
; NO-ATOMIC32-LABEL: store16:
; NO-ATOMIC32:       @ %bb.0:
; NO-ATOMIC32-NEXT:    .save {r7, lr}
; NO-ATOMIC32-NEXT:    push {r7, lr}
; NO-ATOMIC32-NEXT:    movs r1, #0
; NO-ATOMIC32-NEXT:    movs r2, #5
; NO-ATOMIC32-NEXT:    bl __atomic_store_2
; NO-ATOMIC32-NEXT:    pop {r7, pc}
;
; ATOMIC32-LABEL: store16:
; ATOMIC32:       @ %bb.0:
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    movs r1, #0
; ATOMIC32-NEXT:    strh r1, [r0]
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    bx lr
  store atomic i16 0, ptr %p seq_cst, align 2
  ret void
}

define i16 @rmw16(ptr %p) {
; NO-ATOMIC32-LABEL: rmw16:
; NO-ATOMIC32:       @ %bb.0:
; NO-ATOMIC32-NEXT:    .save {r7, lr}
; NO-ATOMIC32-NEXT:    push {r7, lr}
; NO-ATOMIC32-NEXT:    movs r1, #1
; NO-ATOMIC32-NEXT:    movs r2, #5
; NO-ATOMIC32-NEXT:    bl __atomic_fetch_add_2
; NO-ATOMIC32-NEXT:    pop {r7, pc}
;
; ATOMIC32-LABEL: rmw16:
; ATOMIC32:       @ %bb.0:
; ATOMIC32-NEXT:    .save {r7, lr}
; ATOMIC32-NEXT:    push {r7, lr}
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    movs r1, #1
; ATOMIC32-NEXT:    bl __sync_fetch_and_add_2
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    pop {r7, pc}
  %v = atomicrmw add ptr %p, i16 1 seq_cst, align 2
  ret i16 %v
}

define i16 @cmpxchg16(ptr %p) {
; NO-ATOMIC32-LABEL: cmpxchg16:
; NO-ATOMIC32:       @ %bb.0:
; NO-ATOMIC32-NEXT:    .save {r7, lr}
; NO-ATOMIC32-NEXT:    push {r7, lr}
; NO-ATOMIC32-NEXT:    .pad #8
; NO-ATOMIC32-NEXT:    sub sp, #8
; NO-ATOMIC32-NEXT:    add r1, sp, #4
; NO-ATOMIC32-NEXT:    movs r2, #0
; NO-ATOMIC32-NEXT:    strh r2, [r1]
; NO-ATOMIC32-NEXT:    movs r3, #5
; NO-ATOMIC32-NEXT:    str r3, [sp]
; NO-ATOMIC32-NEXT:    movs r2, #1
; NO-ATOMIC32-NEXT:    bl __atomic_compare_exchange_2
; NO-ATOMIC32-NEXT:    ldr r0, [sp, #4]
; NO-ATOMIC32-NEXT:    add sp, #8
; NO-ATOMIC32-NEXT:    pop {r7, pc}
;
; ATOMIC32-LABEL: cmpxchg16:
; ATOMIC32:       @ %bb.0:
; ATOMIC32-NEXT:    .save {r7, lr}
; ATOMIC32-NEXT:    push {r7, lr}
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    movs r1, #0
; ATOMIC32-NEXT:    movs r2, #1
; ATOMIC32-NEXT:    bl __sync_val_compare_and_swap_2
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    pop {r7, pc}
  %res = cmpxchg ptr %p, i16 0, i16 1 seq_cst seq_cst
  %res.0 = extractvalue { i16, i1 } %res, 0
  ret i16 %res.0
}

define i32 @load32(ptr %p) {
; NO-ATOMIC32-LABEL: load32:
; NO-ATOMIC32:       @ %bb.0:
; NO-ATOMIC32-NEXT:    .save {r7, lr}
; NO-ATOMIC32-NEXT:    push {r7, lr}
; NO-ATOMIC32-NEXT:    movs r1, #5
; NO-ATOMIC32-NEXT:    bl __atomic_load_4
; NO-ATOMIC32-NEXT:    pop {r7, pc}
;
; ATOMIC32-LABEL: load32:
; ATOMIC32:       @ %bb.0:
; ATOMIC32-NEXT:    ldr r0, [r0]
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    bx lr
  %v = load atomic i32, ptr %p seq_cst, align 4
  ret i32 %v
}

define void @store32(ptr %p) {
; NO-ATOMIC32-LABEL: store32:
; NO-ATOMIC32:       @ %bb.0:
; NO-ATOMIC32-NEXT:    .save {r7, lr}
; NO-ATOMIC32-NEXT:    push {r7, lr}
; NO-ATOMIC32-NEXT:    movs r1, #0
; NO-ATOMIC32-NEXT:    movs r2, #5
; NO-ATOMIC32-NEXT:    bl __atomic_store_4
; NO-ATOMIC32-NEXT:    pop {r7, pc}
;
; ATOMIC32-LABEL: store32:
; ATOMIC32:       @ %bb.0:
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    movs r1, #0
; ATOMIC32-NEXT:    str r1, [r0]
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    bx lr
  store atomic i32 0, ptr %p seq_cst, align 4
  ret void
}

define i32 @rmw32(ptr %p) {
; NO-ATOMIC32-LABEL: rmw32:
; NO-ATOMIC32:       @ %bb.0:
; NO-ATOMIC32-NEXT:    .save {r7, lr}
; NO-ATOMIC32-NEXT:    push {r7, lr}
; NO-ATOMIC32-NEXT:    movs r1, #1
; NO-ATOMIC32-NEXT:    movs r2, #5
; NO-ATOMIC32-NEXT:    bl __atomic_fetch_add_4
; NO-ATOMIC32-NEXT:    pop {r7, pc}
;
; ATOMIC32-LABEL: rmw32:
; ATOMIC32:       @ %bb.0:
; ATOMIC32-NEXT:    .save {r7, lr}
; ATOMIC32-NEXT:    push {r7, lr}
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    movs r1, #1
; ATOMIC32-NEXT:    bl __sync_fetch_and_add_4
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    pop {r7, pc}
  %v = atomicrmw add ptr %p, i32 1 seq_cst, align 4
  ret i32 %v
}

define i32 @cmpxchg32(ptr %p) {
; NO-ATOMIC32-LABEL: cmpxchg32:
; NO-ATOMIC32:       @ %bb.0:
; NO-ATOMIC32-NEXT:    .save {r7, lr}
; NO-ATOMIC32-NEXT:    push {r7, lr}
; NO-ATOMIC32-NEXT:    .pad #8
; NO-ATOMIC32-NEXT:    sub sp, #8
; NO-ATOMIC32-NEXT:    movs r1, #0
; NO-ATOMIC32-NEXT:    str r1, [sp, #4]
; NO-ATOMIC32-NEXT:    movs r3, #5
; NO-ATOMIC32-NEXT:    str r3, [sp]
; NO-ATOMIC32-NEXT:    add r1, sp, #4
; NO-ATOMIC32-NEXT:    movs r2, #1
; NO-ATOMIC32-NEXT:    bl __atomic_compare_exchange_4
; NO-ATOMIC32-NEXT:    ldr r0, [sp, #4]
; NO-ATOMIC32-NEXT:    add sp, #8
; NO-ATOMIC32-NEXT:    pop {r7, pc}
;
; ATOMIC32-LABEL: cmpxchg32:
; ATOMIC32:       @ %bb.0:
; ATOMIC32-NEXT:    .save {r7, lr}
; ATOMIC32-NEXT:    push {r7, lr}
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    movs r1, #0
; ATOMIC32-NEXT:    movs r2, #1
; ATOMIC32-NEXT:    bl __sync_val_compare_and_swap_4
; ATOMIC32-NEXT:    dmb sy
; ATOMIC32-NEXT:    pop {r7, pc}
  %res = cmpxchg ptr %p, i32 0, i32 1 seq_cst seq_cst
  %res.0 = extractvalue { i32, i1 } %res, 0
  ret i32 %res.0
}

define i64 @load64(ptr %p) {
; CHECK-LABEL: load64:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    .save {r7, lr}
; CHECK-NEXT:    push {r7, lr}
; CHECK-NEXT:    movs r1, #5
; CHECK-NEXT:    bl __atomic_load_8
; CHECK-NEXT:    pop {r7, pc}
  %v = load atomic i64, ptr %p seq_cst, align 8
  ret i64 %v
}

define void @store64(ptr %p) {
; CHECK-LABEL: store64:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    .save {r7, lr}
; CHECK-NEXT:    push {r7, lr}
; CHECK-NEXT:    .pad #8
; CHECK-NEXT:    sub sp, #8
; CHECK-NEXT:    movs r1, #5
; CHECK-NEXT:    str r1, [sp]
; CHECK-NEXT:    movs r2, #0
; CHECK-NEXT:    mov r3, r2
; CHECK-NEXT:    bl __atomic_store_8
; CHECK-NEXT:    add sp, #8
; CHECK-NEXT:    pop {r7, pc}
  store atomic i64 0, ptr %p seq_cst, align 8
  ret void
}

define i64 @rmw64(ptr %p) {
; CHECK-LABEL: rmw64:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    .save {r7, lr}
; CHECK-NEXT:    push {r7, lr}
; CHECK-NEXT:    .pad #8
; CHECK-NEXT:    sub sp, #8
; CHECK-NEXT:    movs r1, #5
; CHECK-NEXT:    str r1, [sp]
; CHECK-NEXT:    movs r2, #1
; CHECK-NEXT:    movs r3, #0
; CHECK-NEXT:    bl __atomic_fetch_add_8
; CHECK-NEXT:    add sp, #8
; CHECK-NEXT:    pop {r7, pc}
  %v = atomicrmw add ptr %p, i64 1 seq_cst, align 8
  ret i64 %v
}

define i64 @cmpxchg64(ptr %p) {
; CHECK-LABEL: cmpxchg64:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    .save {r7, lr}
; CHECK-NEXT:    push {r7, lr}
; CHECK-NEXT:    .pad #16
; CHECK-NEXT:    sub sp, #16
; CHECK-NEXT:    movs r3, #0
; CHECK-NEXT:    str r3, [sp, #12]
; CHECK-NEXT:    str r3, [sp, #8]
; CHECK-NEXT:    movs r1, #5
; CHECK-NEXT:    str r1, [sp]
; CHECK-NEXT:    str r1, [sp, #4]
; CHECK-NEXT:    add r1, sp, #8
; CHECK-NEXT:    movs r2, #1
; CHECK-NEXT:    bl __atomic_compare_exchange_8
; CHECK-NEXT:    ldr r1, [sp, #12]
; CHECK-NEXT:    ldr r0, [sp, #8]
; CHECK-NEXT:    add sp, #16
; CHECK-NEXT:    pop {r7, pc}
  %res = cmpxchg ptr %p, i64 0, i64 1 seq_cst seq_cst
  %res.0 = extractvalue { i64, i1 } %res, 0
  ret i64 %res.0
}
