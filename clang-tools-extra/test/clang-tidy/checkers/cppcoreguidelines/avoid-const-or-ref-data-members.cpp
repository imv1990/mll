// RUN: %check_clang_tidy %s cppcoreguidelines-avoid-const-or-ref-data-members %t
namespace std {
template <typename T>
struct unique_ptr {};

template <typename T>
struct shared_ptr {};
} // namespace std

namespace gsl {
template <typename T>
struct not_null {};
} // namespace gsl

struct Ok {
  int i;
  int *p;
  const int *pc;
  std::unique_ptr<int> up;
  std::shared_ptr<int> sp;
  gsl::not_null<int> n;
};

struct ConstMember {
  const int c;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: member 'c' of type 'const int' is const qualified [cppcoreguidelines-avoid-const-or-ref-data-members]
};

struct LvalueRefMember {
  int &lr;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: member 'lr' of type 'int &' is a reference
};

struct ConstRefMember {
  const int &cr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: member 'cr' of type 'const int &' is a reference
};

struct RvalueRefMember {
  int &&rr;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: member 'rr' of type 'int &&' is a reference
};

struct ConstAndRefMembers {
  const int c;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: member 'c' of type 'const int' is const qualified
  int &lr;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: member 'lr' of type 'int &' is a reference
  const int &cr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: member 'cr' of type 'const int &' is a reference
  int &&rr;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: member 'rr' of type 'int &&' is a reference
};

struct Foo {};

struct Ok2 {
  Foo i;
  Foo *p;
  const Foo *pc;
  std::unique_ptr<Foo> up;
  std::shared_ptr<Foo> sp;
  gsl::not_null<Foo> n;
};

struct ConstMember2 {
  const Foo c;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: member 'c' of type 'const Foo' is const qualified
};

struct LvalueRefMember2 {
  Foo &lr;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: member 'lr' of type 'Foo &' is a reference
};

struct ConstRefMember2 {
  const Foo &cr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: member 'cr' of type 'const Foo &' is a reference
};

struct RvalueRefMember2 {
  Foo &&rr;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: member 'rr' of type 'Foo &&' is a reference
};

struct ConstAndRefMembers2 {
  const Foo c;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: member 'c' of type 'const Foo' is const qualified
  Foo &lr;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: member 'lr' of type 'Foo &' is a reference
  const Foo &cr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: member 'cr' of type 'const Foo &' is a reference
  Foo &&rr;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: member 'rr' of type 'Foo &&' is a reference
};

using ConstType = const int;
using RefType = int &;
using ConstRefType = const int &;
using RefRefType = int &&;

struct WithAlias {
  ConstType c;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: member 'c' of type 'ConstType' (aka 'const int') is const qualified
  RefType lr;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: member 'lr' of type 'RefType' (aka 'int &') is a reference
  ConstRefType cr;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: member 'cr' of type 'ConstRefType' (aka 'const int &') is a reference
  RefRefType rr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: member 'rr' of type 'RefRefType' (aka 'int &&') is a reference
};

template <int N>
using Array = int[N];

struct ConstArrayMember {
  const Array<1> c;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: member 'c' of type 'const Array<1>' (aka 'const int[1]') is const qualified
};

struct LvalueRefArrayMember {
  Array<2> &lr;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: member 'lr' of type 'Array<2> &' (aka 'int (&)[2]') is a reference
};

struct ConstLvalueRefArrayMember {
  const Array<3> &cr;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: member 'cr' of type 'const Array<3> &' (aka 'const int (&)[3]') is a reference
};

struct RvalueRefArrayMember {
  Array<4> &&rr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: member 'rr' of type 'Array<4> &&' (aka 'int (&&)[4]') is a reference
};

template <typename T>
struct TemplatedOk {
  T t;
};

template <typename T>
struct TemplatedConst {
  T t;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: member 't' of type 'const int' is const qualified
};

template <typename T>
struct TemplatedConstRef {
  T t;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: member 't' of type 'const int &' is a reference
};

template <typename T>
struct TemplatedRefRef {
  T t;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: member 't' of type 'int &&' is a reference
};

template <typename T>
struct TemplatedRef {
  T t;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: member 't' of type 'int &' is a reference
};

TemplatedOk<int> t1{};
TemplatedConst<const int> t2{123};
TemplatedConstRef<const int &> t3{123};
TemplatedRefRef<int &&> t4{123};
TemplatedRef<int &> t5{t1.t};
