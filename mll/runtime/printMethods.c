#include <stdio.h>

extern void printStr(const char *c) { printf("%s ", c); }

extern void print_i32(int val) { printf("%d ", val); }

extern void print_i1(unsigned char val) {
  if (val == 1) {
    printf("True ");
  } else {
    printf("False ");
  }
}

extern void print_f32(float val) { printf("%f ", val); }

extern void printSpace() { printf(" "); }

extern void printNewLine() { printf("\n"); }
