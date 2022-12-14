; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv32 -code-model=small -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV32I-SMALL
; RUN: llc -mtriple=riscv32 -code-model=medium -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV32I-MEDIUM
; RUN: llc -mtriple=riscv64 -code-model=small -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV64I-SMALL
; RUN: llc -mtriple=riscv64 -code-model=medium -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV64I-MEDIUM

define void @below_threshold(i32 %in, i32* %out) nounwind {
; RV32I-SMALL-LABEL: below_threshold:
; RV32I-SMALL:       # %bb.0: # %entry
; RV32I-SMALL-NEXT:    li a2, 2
; RV32I-SMALL-NEXT:    blt a2, a0, .LBB0_4
; RV32I-SMALL-NEXT:  # %bb.1: # %entry
; RV32I-SMALL-NEXT:    li a2, 1
; RV32I-SMALL-NEXT:    beq a0, a2, .LBB0_7
; RV32I-SMALL-NEXT:  # %bb.2: # %entry
; RV32I-SMALL-NEXT:    li a2, 2
; RV32I-SMALL-NEXT:    bne a0, a2, .LBB0_10
; RV32I-SMALL-NEXT:  # %bb.3: # %bb2
; RV32I-SMALL-NEXT:    li a0, 3
; RV32I-SMALL-NEXT:    j .LBB0_9
; RV32I-SMALL-NEXT:  .LBB0_4: # %entry
; RV32I-SMALL-NEXT:    li a2, 3
; RV32I-SMALL-NEXT:    beq a0, a2, .LBB0_8
; RV32I-SMALL-NEXT:  # %bb.5: # %entry
; RV32I-SMALL-NEXT:    li a2, 4
; RV32I-SMALL-NEXT:    bne a0, a2, .LBB0_10
; RV32I-SMALL-NEXT:  # %bb.6: # %bb4
; RV32I-SMALL-NEXT:    li a0, 1
; RV32I-SMALL-NEXT:    j .LBB0_9
; RV32I-SMALL-NEXT:  .LBB0_7: # %bb1
; RV32I-SMALL-NEXT:    li a0, 4
; RV32I-SMALL-NEXT:    j .LBB0_9
; RV32I-SMALL-NEXT:  .LBB0_8: # %bb3
; RV32I-SMALL-NEXT:    li a0, 2
; RV32I-SMALL-NEXT:  .LBB0_9: # %exit
; RV32I-SMALL-NEXT:    sw a0, 0(a1)
; RV32I-SMALL-NEXT:  .LBB0_10: # %exit
; RV32I-SMALL-NEXT:    ret
;
; RV32I-MEDIUM-LABEL: below_threshold:
; RV32I-MEDIUM:       # %bb.0: # %entry
; RV32I-MEDIUM-NEXT:    li a2, 2
; RV32I-MEDIUM-NEXT:    blt a2, a0, .LBB0_4
; RV32I-MEDIUM-NEXT:  # %bb.1: # %entry
; RV32I-MEDIUM-NEXT:    li a2, 1
; RV32I-MEDIUM-NEXT:    beq a0, a2, .LBB0_7
; RV32I-MEDIUM-NEXT:  # %bb.2: # %entry
; RV32I-MEDIUM-NEXT:    li a2, 2
; RV32I-MEDIUM-NEXT:    bne a0, a2, .LBB0_10
; RV32I-MEDIUM-NEXT:  # %bb.3: # %bb2
; RV32I-MEDIUM-NEXT:    li a0, 3
; RV32I-MEDIUM-NEXT:    j .LBB0_9
; RV32I-MEDIUM-NEXT:  .LBB0_4: # %entry
; RV32I-MEDIUM-NEXT:    li a2, 3
; RV32I-MEDIUM-NEXT:    beq a0, a2, .LBB0_8
; RV32I-MEDIUM-NEXT:  # %bb.5: # %entry
; RV32I-MEDIUM-NEXT:    li a2, 4
; RV32I-MEDIUM-NEXT:    bne a0, a2, .LBB0_10
; RV32I-MEDIUM-NEXT:  # %bb.6: # %bb4
; RV32I-MEDIUM-NEXT:    li a0, 1
; RV32I-MEDIUM-NEXT:    j .LBB0_9
; RV32I-MEDIUM-NEXT:  .LBB0_7: # %bb1
; RV32I-MEDIUM-NEXT:    li a0, 4
; RV32I-MEDIUM-NEXT:    j .LBB0_9
; RV32I-MEDIUM-NEXT:  .LBB0_8: # %bb3
; RV32I-MEDIUM-NEXT:    li a0, 2
; RV32I-MEDIUM-NEXT:  .LBB0_9: # %exit
; RV32I-MEDIUM-NEXT:    sw a0, 0(a1)
; RV32I-MEDIUM-NEXT:  .LBB0_10: # %exit
; RV32I-MEDIUM-NEXT:    ret
;
; RV64I-SMALL-LABEL: below_threshold:
; RV64I-SMALL:       # %bb.0: # %entry
; RV64I-SMALL-NEXT:    sext.w a0, a0
; RV64I-SMALL-NEXT:    li a2, 2
; RV64I-SMALL-NEXT:    blt a2, a0, .LBB0_4
; RV64I-SMALL-NEXT:  # %bb.1: # %entry
; RV64I-SMALL-NEXT:    li a2, 1
; RV64I-SMALL-NEXT:    beq a0, a2, .LBB0_7
; RV64I-SMALL-NEXT:  # %bb.2: # %entry
; RV64I-SMALL-NEXT:    li a2, 2
; RV64I-SMALL-NEXT:    bne a0, a2, .LBB0_10
; RV64I-SMALL-NEXT:  # %bb.3: # %bb2
; RV64I-SMALL-NEXT:    li a0, 3
; RV64I-SMALL-NEXT:    j .LBB0_9
; RV64I-SMALL-NEXT:  .LBB0_4: # %entry
; RV64I-SMALL-NEXT:    li a2, 3
; RV64I-SMALL-NEXT:    beq a0, a2, .LBB0_8
; RV64I-SMALL-NEXT:  # %bb.5: # %entry
; RV64I-SMALL-NEXT:    li a2, 4
; RV64I-SMALL-NEXT:    bne a0, a2, .LBB0_10
; RV64I-SMALL-NEXT:  # %bb.6: # %bb4
; RV64I-SMALL-NEXT:    li a0, 1
; RV64I-SMALL-NEXT:    j .LBB0_9
; RV64I-SMALL-NEXT:  .LBB0_7: # %bb1
; RV64I-SMALL-NEXT:    li a0, 4
; RV64I-SMALL-NEXT:    j .LBB0_9
; RV64I-SMALL-NEXT:  .LBB0_8: # %bb3
; RV64I-SMALL-NEXT:    li a0, 2
; RV64I-SMALL-NEXT:  .LBB0_9: # %exit
; RV64I-SMALL-NEXT:    sw a0, 0(a1)
; RV64I-SMALL-NEXT:  .LBB0_10: # %exit
; RV64I-SMALL-NEXT:    ret
;
; RV64I-MEDIUM-LABEL: below_threshold:
; RV64I-MEDIUM:       # %bb.0: # %entry
; RV64I-MEDIUM-NEXT:    sext.w a0, a0
; RV64I-MEDIUM-NEXT:    li a2, 2
; RV64I-MEDIUM-NEXT:    blt a2, a0, .LBB0_4
; RV64I-MEDIUM-NEXT:  # %bb.1: # %entry
; RV64I-MEDIUM-NEXT:    li a2, 1
; RV64I-MEDIUM-NEXT:    beq a0, a2, .LBB0_7
; RV64I-MEDIUM-NEXT:  # %bb.2: # %entry
; RV64I-MEDIUM-NEXT:    li a2, 2
; RV64I-MEDIUM-NEXT:    bne a0, a2, .LBB0_10
; RV64I-MEDIUM-NEXT:  # %bb.3: # %bb2
; RV64I-MEDIUM-NEXT:    li a0, 3
; RV64I-MEDIUM-NEXT:    j .LBB0_9
; RV64I-MEDIUM-NEXT:  .LBB0_4: # %entry
; RV64I-MEDIUM-NEXT:    li a2, 3
; RV64I-MEDIUM-NEXT:    beq a0, a2, .LBB0_8
; RV64I-MEDIUM-NEXT:  # %bb.5: # %entry
; RV64I-MEDIUM-NEXT:    li a2, 4
; RV64I-MEDIUM-NEXT:    bne a0, a2, .LBB0_10
; RV64I-MEDIUM-NEXT:  # %bb.6: # %bb4
; RV64I-MEDIUM-NEXT:    li a0, 1
; RV64I-MEDIUM-NEXT:    j .LBB0_9
; RV64I-MEDIUM-NEXT:  .LBB0_7: # %bb1
; RV64I-MEDIUM-NEXT:    li a0, 4
; RV64I-MEDIUM-NEXT:    j .LBB0_9
; RV64I-MEDIUM-NEXT:  .LBB0_8: # %bb3
; RV64I-MEDIUM-NEXT:    li a0, 2
; RV64I-MEDIUM-NEXT:  .LBB0_9: # %exit
; RV64I-MEDIUM-NEXT:    sw a0, 0(a1)
; RV64I-MEDIUM-NEXT:  .LBB0_10: # %exit
; RV64I-MEDIUM-NEXT:    ret
entry:
  switch i32 %in, label %exit [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
  ]
bb1:
  store i32 4, i32* %out
  br label %exit
bb2:
  store i32 3, i32* %out
  br label %exit
bb3:
  store i32 2, i32* %out
  br label %exit
bb4:
  store i32 1, i32* %out
  br label %exit
exit:
  ret void
}

define void @above_threshold(i32 %in, i32* %out) nounwind {
; RV32I-SMALL-LABEL: above_threshold:
; RV32I-SMALL:       # %bb.0: # %entry
; RV32I-SMALL-NEXT:    addi a0, a0, -1
; RV32I-SMALL-NEXT:    li a2, 5
; RV32I-SMALL-NEXT:    bltu a2, a0, .LBB1_9
; RV32I-SMALL-NEXT:  # %bb.1: # %entry
; RV32I-SMALL-NEXT:    slli a0, a0, 2
; RV32I-SMALL-NEXT:    lui a2, %hi(.LJTI1_0)
; RV32I-SMALL-NEXT:    addi a2, a2, %lo(.LJTI1_0)
; RV32I-SMALL-NEXT:    add a0, a0, a2
; RV32I-SMALL-NEXT:    lw a0, 0(a0)
; RV32I-SMALL-NEXT:    jr a0
; RV32I-SMALL-NEXT:  .LBB1_2: # %bb1
; RV32I-SMALL-NEXT:    li a0, 4
; RV32I-SMALL-NEXT:    j .LBB1_8
; RV32I-SMALL-NEXT:  .LBB1_3: # %bb2
; RV32I-SMALL-NEXT:    li a0, 3
; RV32I-SMALL-NEXT:    j .LBB1_8
; RV32I-SMALL-NEXT:  .LBB1_4: # %bb3
; RV32I-SMALL-NEXT:    li a0, 2
; RV32I-SMALL-NEXT:    j .LBB1_8
; RV32I-SMALL-NEXT:  .LBB1_5: # %bb4
; RV32I-SMALL-NEXT:    li a0, 1
; RV32I-SMALL-NEXT:    j .LBB1_8
; RV32I-SMALL-NEXT:  .LBB1_6: # %bb5
; RV32I-SMALL-NEXT:    li a0, 100
; RV32I-SMALL-NEXT:    j .LBB1_8
; RV32I-SMALL-NEXT:  .LBB1_7: # %bb6
; RV32I-SMALL-NEXT:    li a0, 200
; RV32I-SMALL-NEXT:  .LBB1_8: # %exit
; RV32I-SMALL-NEXT:    sw a0, 0(a1)
; RV32I-SMALL-NEXT:  .LBB1_9: # %exit
; RV32I-SMALL-NEXT:    ret
;
; RV32I-MEDIUM-LABEL: above_threshold:
; RV32I-MEDIUM:       # %bb.0: # %entry
; RV32I-MEDIUM-NEXT:    addi a0, a0, -1
; RV32I-MEDIUM-NEXT:    li a2, 5
; RV32I-MEDIUM-NEXT:    bltu a2, a0, .LBB1_9
; RV32I-MEDIUM-NEXT:  # %bb.1: # %entry
; RV32I-MEDIUM-NEXT:    slli a0, a0, 2
; RV32I-MEDIUM-NEXT:  .Lpcrel_hi0:
; RV32I-MEDIUM-NEXT:    auipc a2, %pcrel_hi(.LJTI1_0)
; RV32I-MEDIUM-NEXT:    addi a2, a2, %pcrel_lo(.Lpcrel_hi0)
; RV32I-MEDIUM-NEXT:    add a0, a0, a2
; RV32I-MEDIUM-NEXT:    lw a0, 0(a0)
; RV32I-MEDIUM-NEXT:    jr a0
; RV32I-MEDIUM-NEXT:  .LBB1_2: # %bb1
; RV32I-MEDIUM-NEXT:    li a0, 4
; RV32I-MEDIUM-NEXT:    j .LBB1_8
; RV32I-MEDIUM-NEXT:  .LBB1_3: # %bb2
; RV32I-MEDIUM-NEXT:    li a0, 3
; RV32I-MEDIUM-NEXT:    j .LBB1_8
; RV32I-MEDIUM-NEXT:  .LBB1_4: # %bb3
; RV32I-MEDIUM-NEXT:    li a0, 2
; RV32I-MEDIUM-NEXT:    j .LBB1_8
; RV32I-MEDIUM-NEXT:  .LBB1_5: # %bb4
; RV32I-MEDIUM-NEXT:    li a0, 1
; RV32I-MEDIUM-NEXT:    j .LBB1_8
; RV32I-MEDIUM-NEXT:  .LBB1_6: # %bb5
; RV32I-MEDIUM-NEXT:    li a0, 100
; RV32I-MEDIUM-NEXT:    j .LBB1_8
; RV32I-MEDIUM-NEXT:  .LBB1_7: # %bb6
; RV32I-MEDIUM-NEXT:    li a0, 200
; RV32I-MEDIUM-NEXT:  .LBB1_8: # %exit
; RV32I-MEDIUM-NEXT:    sw a0, 0(a1)
; RV32I-MEDIUM-NEXT:  .LBB1_9: # %exit
; RV32I-MEDIUM-NEXT:    ret
;
; RV64I-SMALL-LABEL: above_threshold:
; RV64I-SMALL:       # %bb.0: # %entry
; RV64I-SMALL-NEXT:    sext.w a0, a0
; RV64I-SMALL-NEXT:    addi a0, a0, -1
; RV64I-SMALL-NEXT:    li a2, 5
; RV64I-SMALL-NEXT:    bltu a2, a0, .LBB1_9
; RV64I-SMALL-NEXT:  # %bb.1: # %entry
; RV64I-SMALL-NEXT:    slli a0, a0, 2
; RV64I-SMALL-NEXT:    lui a2, %hi(.LJTI1_0)
; RV64I-SMALL-NEXT:    addi a2, a2, %lo(.LJTI1_0)
; RV64I-SMALL-NEXT:    add a0, a0, a2
; RV64I-SMALL-NEXT:    lw a0, 0(a0)
; RV64I-SMALL-NEXT:    jr a0
; RV64I-SMALL-NEXT:  .LBB1_2: # %bb1
; RV64I-SMALL-NEXT:    li a0, 4
; RV64I-SMALL-NEXT:    j .LBB1_8
; RV64I-SMALL-NEXT:  .LBB1_3: # %bb2
; RV64I-SMALL-NEXT:    li a0, 3
; RV64I-SMALL-NEXT:    j .LBB1_8
; RV64I-SMALL-NEXT:  .LBB1_4: # %bb3
; RV64I-SMALL-NEXT:    li a0, 2
; RV64I-SMALL-NEXT:    j .LBB1_8
; RV64I-SMALL-NEXT:  .LBB1_5: # %bb4
; RV64I-SMALL-NEXT:    li a0, 1
; RV64I-SMALL-NEXT:    j .LBB1_8
; RV64I-SMALL-NEXT:  .LBB1_6: # %bb5
; RV64I-SMALL-NEXT:    li a0, 100
; RV64I-SMALL-NEXT:    j .LBB1_8
; RV64I-SMALL-NEXT:  .LBB1_7: # %bb6
; RV64I-SMALL-NEXT:    li a0, 200
; RV64I-SMALL-NEXT:  .LBB1_8: # %exit
; RV64I-SMALL-NEXT:    sw a0, 0(a1)
; RV64I-SMALL-NEXT:  .LBB1_9: # %exit
; RV64I-SMALL-NEXT:    ret
;
; RV64I-MEDIUM-LABEL: above_threshold:
; RV64I-MEDIUM:       # %bb.0: # %entry
; RV64I-MEDIUM-NEXT:    sext.w a0, a0
; RV64I-MEDIUM-NEXT:    addi a0, a0, -1
; RV64I-MEDIUM-NEXT:    li a2, 5
; RV64I-MEDIUM-NEXT:    bltu a2, a0, .LBB1_9
; RV64I-MEDIUM-NEXT:  # %bb.1: # %entry
; RV64I-MEDIUM-NEXT:    slli a0, a0, 3
; RV64I-MEDIUM-NEXT:  .Lpcrel_hi0:
; RV64I-MEDIUM-NEXT:    auipc a2, %pcrel_hi(.LJTI1_0)
; RV64I-MEDIUM-NEXT:    addi a2, a2, %pcrel_lo(.Lpcrel_hi0)
; RV64I-MEDIUM-NEXT:    add a0, a0, a2
; RV64I-MEDIUM-NEXT:    ld a0, 0(a0)
; RV64I-MEDIUM-NEXT:    jr a0
; RV64I-MEDIUM-NEXT:  .LBB1_2: # %bb1
; RV64I-MEDIUM-NEXT:    li a0, 4
; RV64I-MEDIUM-NEXT:    j .LBB1_8
; RV64I-MEDIUM-NEXT:  .LBB1_3: # %bb2
; RV64I-MEDIUM-NEXT:    li a0, 3
; RV64I-MEDIUM-NEXT:    j .LBB1_8
; RV64I-MEDIUM-NEXT:  .LBB1_4: # %bb3
; RV64I-MEDIUM-NEXT:    li a0, 2
; RV64I-MEDIUM-NEXT:    j .LBB1_8
; RV64I-MEDIUM-NEXT:  .LBB1_5: # %bb4
; RV64I-MEDIUM-NEXT:    li a0, 1
; RV64I-MEDIUM-NEXT:    j .LBB1_8
; RV64I-MEDIUM-NEXT:  .LBB1_6: # %bb5
; RV64I-MEDIUM-NEXT:    li a0, 100
; RV64I-MEDIUM-NEXT:    j .LBB1_8
; RV64I-MEDIUM-NEXT:  .LBB1_7: # %bb6
; RV64I-MEDIUM-NEXT:    li a0, 200
; RV64I-MEDIUM-NEXT:  .LBB1_8: # %exit
; RV64I-MEDIUM-NEXT:    sw a0, 0(a1)
; RV64I-MEDIUM-NEXT:  .LBB1_9: # %exit
; RV64I-MEDIUM-NEXT:    ret
entry:
  switch i32 %in, label %exit [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
    i32 5, label %bb5
    i32 6, label %bb6
  ]
bb1:
  store i32 4, i32* %out
  br label %exit
bb2:
  store i32 3, i32* %out
  br label %exit
bb3:
  store i32 2, i32* %out
  br label %exit
bb4:
  store i32 1, i32* %out
  br label %exit
bb5:
  store i32 100, i32* %out
  br label %exit
bb6:
  store i32 200, i32* %out
  br label %exit
exit:
  ret void
}
