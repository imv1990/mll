; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s -check-prefix=X86-64
; RUN: llc < %s -mtriple=x86_64-cygwin | FileCheck %s -check-prefix=WIN64
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s -check-prefix=WIN64
; RUN: llc < %s -mtriple=x86_64-mingw32 | FileCheck %s -check-prefix=WIN64

define i64 @mod128(i128 %x) nounwind {
; X86-64-LABEL: mod128:
; X86-64:       # %bb.0:
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $3, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __modti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: mod128:
; WIN64:       # %bb.0:
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $3, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __modti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq


  %1 = srem i128 %x, 3
  %2 = trunc i128 %1 to i64
  ret i64 %2
}

define i64 @div128(i128 %x) nounwind {
; X86-64-LABEL: div128:
; X86-64:       # %bb.0:
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $3, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __divti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: div128:
; WIN64:       # %bb.0:
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $3, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __divti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq


  %1 = sdiv i128 %x, 3
  %2 = trunc i128 %1 to i64
  ret i64 %2
}

define i64 @umod128(i128 %x) nounwind {
; X86-64-LABEL: umod128:
; X86-64:       # %bb.0:
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $3, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __umodti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: umod128:
; WIN64:       # %bb.0:
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $3, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __umodti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq


  %1 = urem i128 %x, 3
  %2 = trunc i128 %1 to i64
  ret i64 %2
}

define i64 @udiv128(i128 %x) nounwind {
; X86-64-LABEL: udiv128:
; X86-64:       # %bb.0:
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $3, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __udivti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: udiv128:
; WIN64:       # %bb.0:
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $3, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __udivti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq


  %1 = udiv i128 %x, 3
  %2 = trunc i128 %1 to i64
  ret i64 %2
}

define i128 @urem_i128_3(i128 %x) nounwind {
; X86-64-LABEL: urem_i128_3:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $3, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __umodti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: urem_i128_3:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $3, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __umodti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = urem i128 %x, 3
  ret i128 %rem
}

define i128 @urem_i128_5(i128 %x) nounwind {
; X86-64-LABEL: urem_i128_5:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $5, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __umodti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: urem_i128_5:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $5, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __umodti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = urem i128 %x, 5
  ret i128 %rem
}

define i128 @urem_i128_15(i128 %x) nounwind {
; X86-64-LABEL: urem_i128_15:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $15, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __umodti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: urem_i128_15:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $15, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __umodti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = urem i128 %x, 15
  ret i128 %rem
}

define i128 @urem_i128_17(i128 %x) nounwind {
; X86-64-LABEL: urem_i128_17:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $17, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __umodti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: urem_i128_17:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $17, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __umodti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = urem i128 %x, 17
  ret i128 %rem
}

define i128 @urem_i128_255(i128 %x) nounwind {
; X86-64-LABEL: urem_i128_255:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $255, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __umodti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: urem_i128_255:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $255, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __umodti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = urem i128 %x, 255
  ret i128 %rem
}

define i128 @urem_i128_257(i128 %x) nounwind {
; X86-64-LABEL: urem_i128_257:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $257, %edx # imm = 0x101
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __umodti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: urem_i128_257:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $257, {{[0-9]+}}(%rsp) # imm = 0x101
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __umodti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = urem i128 %x, 257
  ret i128 %rem
}

define i128 @urem_i128_65535(i128 %x) nounwind {
; X86-64-LABEL: urem_i128_65535:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $65535, %edx # imm = 0xFFFF
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __umodti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: urem_i128_65535:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $65535, {{[0-9]+}}(%rsp) # imm = 0xFFFF
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __umodti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = urem i128 %x, 65535
  ret i128 %rem
}

define i128 @urem_i128_65537(i128 %x) nounwind {
; X86-64-LABEL: urem_i128_65537:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $65537, %edx # imm = 0x10001
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __umodti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: urem_i128_65537:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $65537, {{[0-9]+}}(%rsp) # imm = 0x10001
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __umodti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = urem i128 %x, 65537
  ret i128 %rem
}

define i128 @urem_i128_12(i128 %x) nounwind {
; X86-64-LABEL: urem_i128_12:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $12, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __umodti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: urem_i128_12:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $12, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __umodti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = urem i128 %x, 12
  ret i128 %rem
}

define i128 @udiv_i128_3(i128 %x) nounwind {
; X86-64-LABEL: udiv_i128_3:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $3, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __udivti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: udiv_i128_3:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $3, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __udivti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = udiv i128 %x, 3
  ret i128 %rem
}

define i128 @udiv_i128_5(i128 %x) nounwind {
; X86-64-LABEL: udiv_i128_5:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $5, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __udivti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: udiv_i128_5:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $5, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __udivti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = udiv i128 %x, 5
  ret i128 %rem
}

define i128 @udiv_i128_15(i128 %x) nounwind {
; X86-64-LABEL: udiv_i128_15:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $15, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __udivti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: udiv_i128_15:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $15, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __udivti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = udiv i128 %x, 15
  ret i128 %rem
}

define i128 @udiv_i128_17(i128 %x) nounwind {
; X86-64-LABEL: udiv_i128_17:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $17, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __udivti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: udiv_i128_17:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $17, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __udivti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = udiv i128 %x, 17
  ret i128 %rem
}

define i128 @udiv_i128_255(i128 %x) nounwind {
; X86-64-LABEL: udiv_i128_255:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $255, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __udivti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: udiv_i128_255:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $255, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __udivti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = udiv i128 %x, 255
  ret i128 %rem
}

define i128 @udiv_i128_257(i128 %x) nounwind {
; X86-64-LABEL: udiv_i128_257:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $257, %edx # imm = 0x101
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __udivti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: udiv_i128_257:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $257, {{[0-9]+}}(%rsp) # imm = 0x101
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __udivti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = udiv i128 %x, 257
  ret i128 %rem
}

define i128 @udiv_i128_65535(i128 %x) nounwind {
; X86-64-LABEL: udiv_i128_65535:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $65535, %edx # imm = 0xFFFF
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __udivti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: udiv_i128_65535:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $65535, {{[0-9]+}}(%rsp) # imm = 0xFFFF
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __udivti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = udiv i128 %x, 65535
  ret i128 %rem
}

define i128 @udiv_i128_65537(i128 %x) nounwind {
; X86-64-LABEL: udiv_i128_65537:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $65537, %edx # imm = 0x10001
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __udivti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: udiv_i128_65537:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $65537, {{[0-9]+}}(%rsp) # imm = 0x10001
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __udivti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = udiv i128 %x, 65537
  ret i128 %rem
}

define i128 @udiv_i128_12(i128 %x) nounwind {
; X86-64-LABEL: udiv_i128_12:
; X86-64:       # %bb.0: # %entry
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    movl $12, %edx
; X86-64-NEXT:    xorl %ecx, %ecx
; X86-64-NEXT:    callq __udivti3@PLT
; X86-64-NEXT:    popq %rcx
; X86-64-NEXT:    retq
;
; WIN64-LABEL: udiv_i128_12:
; WIN64:       # %bb.0: # %entry
; WIN64-NEXT:    subq $72, %rsp
; WIN64-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $12, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN64-NEXT:    callq __udivti3
; WIN64-NEXT:    movq %xmm0, %rax
; WIN64-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; WIN64-NEXT:    movq %xmm0, %rdx
; WIN64-NEXT:    addq $72, %rsp
; WIN64-NEXT:    retq
entry:
  %rem = udiv i128 %x, 12
  ret i128 %rem
}
