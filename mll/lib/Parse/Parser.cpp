#include "mll/Parse/Parser.h"
#include "mll/AST/AST.h"
#include "mll/AST/BinaryOperator.h"
#include "mll/Parse/BuiltinDialect.h"
#include "mll/Parse/ParserState.h"
#include "mll/Parse/Token.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"
#include <stack>

using namespace mll;
ASTModule *mll::parseInputFile(std::unique_ptr<llvm::MemoryBuffer> file,
                               MLLContext *context) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  llvm::SmallVector<SymbolTable *, 4> symbolList;
  ParserState State(sourceMgr, symbolList, context);

  Parser parser(State, sourceMgr);
  return parser.parseASTModule();
}

bool Parser::isFunctionName(StringRef str) {
  auto sym = getSymbol(str);
  if (!sym)
    return false;

  return llvm::isa<builtin::FunctionType>(sym->getType());
}

/// ast-module ::= statement-list
///
/// ASTModule body is stored as block.
ASTModule *Parser::parseASTModule() {

  auto startLoc = getLoc();
  StmtList nodes;
  auto symTable = builder.create<SymbolTable>("ASTModule", nullptr);
  state.symbols.push_back(symTable);

  while (!getToken().is(Token::eof)) {
    auto stmt = parseStmt();
    if (!stmt) {
      return nullptr;
    }
    nodes.push_back(stmt);
  }

  auto body = builder.create<Block>(startLoc, nodes);
  return builder.create<ASTModule>(startLoc, body, symTable);
}

/// block ::=  `{` statement-list `}`
Block *Parser::parseBlock(SymbolTable *symTable) {
  consumeToken(Token::l_brace);
  pushSymbolTable(symTable);

  StmtList nodes;
  auto startLoc = getLoc();
  while (!getToken().is(Token::r_brace)) {
    auto stmt = parseStmt();
    assert(stmt);
    nodes.push_back(stmt);
  }

  popSymbolTable();
  consumeToken(Token::r_brace);

  return builder.create<Block>(startLoc, nodes);
}

/// statement ::= (((dialaect-name`.`)? statement-name custom-stmt-format) |
///               (function-name '(` function-operand-list `)`) |
///               (assign-stmt))
///
/// assign-stmt ::= lhs-expr-operand `=` rhs-expr-operand
///
/// NOTE: Variable declarations are not handled.
Stmt *Parser::parseStmt() {

  if (isToken(Token::identifier) && isDialectName(getTokenSpelling())) {
    auto dialectName = getTokenSpelling();
    consumeToken();
    // should be like <dialect-name>.<identifier>
    consumeToken(Token::dot);
    assert(isToken(Token::identifier));
    auto id = getTokenSpelling();
    auto dialect = getDialect(dialectName);
    assert(dialect);

    auto node = parseDialectIdentifier(dialect, id);
    auto stmt = llvm::dyn_cast<Stmt>(node);
    assert(stmt);
    return stmt;
  }

  // Process if it is the builtin dialect symbol.
  if (isToken(Token::identifier) && isBuiltinDialectId(getTokenSpelling())) {
    auto id = getTokenSpelling();
    auto node = parseDialectIdentifier(getBuiltinDialect(), id);
    auto stmt = llvm::dyn_cast<Stmt>(node);
    assert(stmt);
    return stmt;
  }

  assert(isToken(Token::identifier));
  auto identifier = getTokenSpelling();
  auto loc = getLoc();
  auto symbol = getSymbol(identifier);

  if (isFunctionName(identifier)) {
    auto expr = parseExpr();
    if (!expr) {
      return nullptr;
    }
    return builder.create<builtin::ExprStmt>(loc, expr);
  }

  // Should be assignment statement
  if (symbol) {
    auto lhs = parseVariableExpr();
    if (!lhs) {
      return nullptr;
    }
    consumeToken(Token::equal);
    auto rhs = parseExpr();
    if (!rhs) {
      return nullptr;
    }
    return builder.create<builtin::AssignStmt>(loc, lhs, rhs);
  }

  // Assignment statement: New variable.
  consumeToken(Token::identifier);
  consumeToken(Token::equal);
  auto rhs = parseExpr();
  if (!rhs) {
    return nullptr;
  }
  symbol = builder.create<Symbol>(identifier, rhs->getExprType(),
                                  state.symbols.back());
  insertSymbol(symbol);
  auto symExpr =
      builder.create<builtin::SymbolExpr>(loc, symbol->getType(), symbol);
  return builder.create<builtin::AssignStmt>(loc, symExpr, rhs);
}

/// expr :=  `(`? expr-operand `)`? |
///          `(`? expr-operand binary-operator expr-operand `)` |
///          `(`? expr binary-operator expr `)`?
///
/// expr-operand := constant-expr |
///                 symbol-expr |
///                 custom-dialect-expr |
///                 complex-symbol-expr
///
/// symbol-expr ::= identifier
///
/// constant-expr ::= integer | floatliteral | custom-constant-expr
///
/// custom-constant-expr ::= type custom-syntax
///
/// custom-dialect-expr := dialect-name`.expr-name custom-syntax
///
/// expr-name ::= identifier
///
/// complex-symbol-expr ::= symbol-expr custom-syntax
///
///
/// binary-operator :=  (`+` | `-` | `*` | `/`)

static bool isConstant(Token tok) {
  switch (tok.getKind()) {
  case Token::integer:
  case Token::floatliteral:
  case Token::string:
    return true;
  default:
    return false;
  }
}

bool Parser::isOperator(std::string identifier) {
  auto binOp = state.context->getBinaryOperator(identifier);
  return binOp != nullptr;
}

int Parser::getOpPrecedence(Parser::OperatorInfo op) {
  /// '(` and ')' has zero precedence
  if (!op.isOperator()) {
    return 0;
  }
  return op.getOpPrecedence();
}

Expr *Parser::parsePropertyExpr(Dialect::PropertyExprInfo &info) {
  assert(isToken(Token::identifier));
  typedef ::mll::Expr *(*propertyExprMethod)(mll::ASTBuilder & builder,
                                             Location loc);
  auto method = (propertyExprMethod)info.buildFn;
  assert(method);
  auto expr = method(builder, getLoc());
  assert(expr && "Failed to build property expr");
  consumeToken();
  return expr;
}

Stmt *Parser::parseCustomStmt(Dialect::StmtInfo &info) {
  assert(isToken(Token::identifier));
  consumeToken();
  typedef ::mll::Stmt *(*stmtParseMethod)(mll::Parser & parser,
                                          mll::ASTBuilder & builder);
  auto method = (stmtParseMethod)info.parserFn;
  assert(method);
  auto stmt = method(*this, builder);
  assert(stmt && "Failed to parse statement info");
  return stmt;
}

Type *Parser::parseType() {
  if (isToken(Token::identifier) && isDialectName(getTokenSpelling())) {
    auto dialectName = getTokenSpelling();
    consumeToken();
    // should be like <dialect-name>.<identifier>
    consumeToken(Token::dot);
    assert(isToken(Token::identifier));
    auto id = getTokenSpelling();
    auto dialect = getDialect(dialectName);
    assert(dialect);

    assert(dialect->getIdentifierKind(id) == Dialect::TypeKind);
    auto typeInfo = dialect->getTypeInfo(id);
    return parseCustomType(typeInfo);
  }

  // Process if it is the builtin dialect symbol.
  if (isToken(Token::identifier) && isBuiltinDialectId(getTokenSpelling())) {
    auto id = getTokenSpelling();

    auto dialect = getBuiltinDialect();
    assert(dialect->getIdentifierKind(id) == Dialect::TypeKind);
    auto typeInfo = dialect->getTypeInfo(id);
    return parseCustomType(typeInfo);
  }

  return nullptr;
}

Type *Parser::parseCustomType(Dialect::TypeInfo &info) {
  assert(isToken(Token::identifier));

  // Parse builtin scalar types here itself.
  auto id = getTokenSpelling();
  consumeToken();
  if (id == "i32") {
    return builder.getType<builtin::I32Type>();
  }
  if (id == "f32") {
    return builder.getType<builtin::F32Type>();
  }
  if (id == "string") {
    return builder.getType<builtin::StringType>();
  }

  typedef ::mll::Type *(*constExprParserFn)(mll::Parser & parser,
                                            mll::ASTBuilder & builder);
  auto method = (constExprParserFn)info.parserFn;
  assert(method);
  auto type = method(*this, builder);
  assert(type && "Failed to parse custom type");
  return type;
}

Expr *Parser::parseConstantExpr(Type *ty) {
  auto dialect = ty->getDialect(state.context);
  auto info = dialect->getConstantExprInfo(ty->getName());
  typedef ::mll::Expr *(*constExprParserFn)(
      mll::Parser & parser, mll::ASTBuilder & builder, Type * ty);
  auto method = (constExprParserFn)info.parserFn;
  assert(method);
  auto expr = method(*this, builder, ty);
  assert(expr && "Failed to parse statement info");
  return expr;
}

Expr *Parser::parseMethodExpr(Dialect::MethodExprInfo &info) {
  assert(isToken(Token::identifier));
  consumeToken();

  consumeToken(Token::l_paren);
  llvm::SmallVector<TypeExprInfo, 2> exprs;

  // TODO: Handle type expressions here itself.
  while (!isToken(Token::r_paren)) {

    // Parse type if it is one of them.
    if (auto ty = parseType()) {
      TypeExprInfo info;
      info.value.type = ty;
      info.isType = true;
      exprs.push_back(info);
    } else {
      auto expr = parseExpr();
      if (!expr) {
        return nullptr;
      }
      TypeExprInfo info;
      info.value.expr = expr;
      info.isType = false;
      exprs.push_back(info);
    }
    if (!isToken(Token::r_paren))
      consumeToken(Token::comma);
  }

  typedef ::mll::Expr *(*methodExprFunc)(mll::ASTBuilder & builder,
                                         Location loc,
                                         llvm::SmallVectorImpl<TypeExprInfo> &);
  auto method = (methodExprFunc)info.buildFn;
  assert(method);
  auto expr = method(builder, getLoc(), exprs);
  assert(expr && "Failed to build property expr");
  consumeToken(Token::r_paren);
  return expr;
}

/// Check if the expression operand is a dialect identifier.
/// Check if it is the type expression like i32.min
ASTNode *Parser::parseDialectIdentifier(Dialect *dialect, StringRef id) {
  auto idKind = dialect->getIdentifierKind(id);
  if (idKind == Dialect::None) {
    emitError(getLoc(), "Unknown dialect symbol");
    return nullptr;
  }

  switch (idKind) {
  case Dialect::PropertyExprKind: {
    auto propExprInfo = dialect->getPropertyExprInfo(id);
    return parsePropertyExpr(propExprInfo);
  }
  case Dialect::MethodExprKind: {
    auto propExprInfo = dialect->getMethodExprInfo(id);
    return parseMethodExpr(propExprInfo);
  }
  case Dialect::StmtKind: {
    auto stmtInfo = dialect->getStmtInfo(id);
    return parseCustomStmt(stmtInfo);
  }

  // Should be constant expression at this point.
  case Dialect::TypeKind: {
    auto typeInfo = dialect->getTypeInfo(id);
    auto ty = parseCustomType(typeInfo);

    assert(isToken(Token::dot));
    // Is a constant expression.
    consumeToken();
    return parseConstantExpr(ty);
  }
  default:
    emitError(getLoc(), "Unhandled dialect symbol");
    return nullptr;
  }
}

Expr *Parser::parseVariableExpr() {
  assert(isToken(Token::identifier));
  auto variable = getTokenSpelling();
  auto loc = getLoc();
  auto symbol = getSymbol(variable);
  if (!symbol) {
    consumeToken();
    emitError(loc, "Variable not defined : ") << variable;
    return nullptr;
  }

  auto type = symbol->getType();
  assert(type);
  auto dialect = type->getDialect(state.context);

  if (!dialect->hasVariableExprInfo(type->getName())) {
    consumeToken();
    return builder.create<builtin::SymbolExpr>(loc, symbol->getType(), symbol);
  }

  auto info = dialect->getVariableExprInfo(type->getName());

  typedef ::mll::Expr *(*constExprParserFn)(
      mll::Parser & parser, mll::ASTBuilder & builder, Type * ty);

  auto method = (constExprParserFn)info.parserFn;
  assert(method);
  auto expr = method(*this, builder, type);
  assert(expr && "Failed to parse Variable Expression info");
  return expr;
}

Expr *Parser::parseExprOperand(Token::Kind prevToken) {

  /// Check if it is any of literal constants like 3, 2.2 , -2 etc.
  if (isToken(Token::integer)) {
    auto constVal = builder.create<builtin::I32ConstantExpr>(
        getLoc(), builder.getType<builtin::I32Type>(),
        getToken().getIntegerValue());
    consumeToken();
    return constVal;
  }

  /// Check if it is any of literal constants like 3, 2.2 , -2 etc.
  if (isToken(Token::floatliteral)) {
    auto constVal = builder.create<builtin::F32ConstantExpr>(
        getLoc(), builder.getType<builtin::F32Type>(),
        getToken().getFloatingPointValue());
    consumeToken();
    return constVal;
  }

  /// Check if it is any of string constants like "a", "aaa", etc.
  if (isToken(Token::string)) {
    auto constVal = builder.create<builtin::StringConstantExpr>(
        getLoc(), builder.getType<builtin::StringType>(),
        getToken().getStringValue());
    consumeToken();
    return constVal;
  }

  // Parse [expr, expr2, ...] list expression.
  if (isToken(Token::l_square)) {
    auto loc = getLoc();
    consumeToken();

    ExprList list;
    while (!isToken(Token::r_square)) {
      auto expr = parseExpr();
      if (!expr) {
        return nullptr;
      }
      list.push_back(expr);
      if (!isToken(Token::r_square)) {
        consumeToken(Token::comma);
      }
    }
    consumeToken(Token::r_square);

    return builder.create<builtin::ListExpr>(
        loc, builder.getType<builtin::ListType>(), list);
  }

  if (isToken(Token::identifier) && isDialectName(getTokenSpelling())) {
    auto dialectName = getTokenSpelling();
    consumeToken();
    // should be like <dialect-name>.<identifier>
    consumeToken(Token::dot);
    assert(isToken(Token::identifier));
    auto id = getTokenSpelling();
    auto dialect = getDialect(dialectName);
    assert(dialect);

    auto node = parseDialectIdentifier(dialect, id);
    auto expr = llvm::dyn_cast<Expr>(node);
    assert(expr);
    return expr;
  }

  // Process if it is the builtin dialect symbol.
  if (isToken(Token::identifier) && isBuiltinDialectId(getTokenSpelling())) {
    auto id = getTokenSpelling();
    auto node = parseDialectIdentifier(getBuiltinDialect(), id);
    auto expr = llvm::dyn_cast<Expr>(node);
    assert(expr);
    return expr;
  }

  // Is a function call.
  if (isToken(Token::identifier) && isFunctionName(getTokenSpelling())) {
    auto funcSym = getSymbol(getTokenSpelling());
    assert(funcSym);
    consumeToken();

    auto funcType = llvm::cast<builtin::FunctionType>(funcSym->getType());

    if (!isToken(Token::l_paren)) {
      emitError(getLoc(), "Expecting ( while parsing function call");
      return nullptr;
    }

    consumeToken(Token::l_paren);

    ExprList list;
    while (!isToken(Token::r_paren)) {
      auto expr = parseExpr();
      if (!expr) {
        return nullptr;
      }
      list.push_back(expr);
      if (!isToken(Token::r_paren)) {
        consumeToken(Token::comma);
      }
    }
    consumeToken(Token::r_paren);

    auto args = builder.create<builtin::ListExpr>(
        getLoc(), builder.getType<builtin::ListType>(), list);

    return builder.create<builtin::CallExpr>(getLoc(), funcType->returnType(),
                                             funcSym, args);
  }

  /// Check if it is variable expression like a.size() or a[2] where a is
  /// variable of type array.
  // Should be a variable
  if (isToken(Token::identifier)) {
    return parseVariableExpr();
  }

  emitError(getLoc(), "Unknown expression type");
  return nullptr;
}

// Build appropriate expression based on \p op.
Expr *Parser::checkAndBuildExpr(Expr *lhs, Expr *rhs, OperatorInfo op,
                                Type *type) {
  return builder.create<builtin::BinaryOpExpr>(lhs->getLoc(), type, lhs, rhs,
                                               op.getOperatorKeyword());
}

// Build the expression and push it to the stack.
void Parser::pushOperation(std::stack<Expr *> &valueStack,
                           std::stack<OperatorInfo> &opsStack, Location loc) {
  auto op = opsStack.top();
  opsStack.pop();
  if (valueStack.empty()) {
    emitError(loc, "Unmatched paranthesis");
    return;
  }

  auto val2 = valueStack.top();
  valueStack.pop();
  Expr *val1 = nullptr;
  if (valueStack.empty()) {
    emitError(loc, "Unmatched paranthesis");
    return;
  }
  val1 = valueStack.top();
  valueStack.pop();

  assert(val1->getExprType() == val2->getExprType());

  auto retType = val1->getExprType();
  if (op.isRelational()) {
    retType = builder.getType<builtin::I1Type>();
  }
  auto expr = checkAndBuildExpr(val1, val2, op, retType);
  valueStack.push(expr);
}

bool Parser::isExprOperand(Token tok) {
  return (!isOperator(tok.getSpelling().str())) &&
         (isConstant(tok) || tok.is(Token::identifier) ||
          tok.is(Token::l_square) || isFunctionName(tok.getSpelling()));
}

Expr *Parser::parseExpr() {

  std::stack<Expr *> valueStack;
  std::stack<OperatorInfo> opsStack;
  int paranBalanced = 0;

  auto loc = getLoc();
  auto prevToken = Token(Token::unknown, "");
  while (true) {
    auto currTok = getToken();
    auto currTokKind = getToken().getKind();

    // All parseable expressions are parsed and
    // single expression is generated.
    if (valueStack.size() == 1 && opsStack.empty() &&
        !isOperator(getTokenSpelling().str())) {
      break;
    }

    if (valueStack.size() == 2 && opsStack.size() == 1 &&
        !isOperator(getTokenSpelling().str())) {
      pushOperation(valueStack, opsStack, valueStack.top()->getLoc());
      break;
    }

    // Process (.
    if (currTok.getKind() == Token::l_paren) {
      opsStack.push(OperatorInfo(Token::l_paren));
      consumeToken(); // '('
      prevToken = currTok;
      paranBalanced++;
      continue;
    }

    // Process all the operands.
    if (isExprOperand(currTok)) {

      // Two expression operands cannot be next to each other
      if (isExprOperand(prevToken)) {
        break;
      }
      auto expr = parseExprOperand(prevToken.getKind());
      if (!expr) {
        return nullptr;
      }
      valueStack.push(expr);
      prevToken = currTok;
      continue;
    }

    // Process ).
    if (currTokKind == Token::r_paren) {
      while (!opsStack.empty() && !opsStack.top().isTokenKind(Token::l_paren)) {
        pushOperation(valueStack, opsStack, getLoc());
      }

      // Consume l_paren.
      if (opsStack.empty()) {
        if (paranBalanced != 0) {
          emitError(getLoc(), "Unmatched paranthesis");
          return nullptr;
        }
        break;
      }
      // should be l_paren now.
      assert(opsStack.top().isTokenKind(Token::l_paren));
      opsStack.pop();
      consumeToken(); // ')'

      // Look if the next token is operator, else we are done with processing.
      if (opsStack.empty()) {
        if (!isOperator(getTokenSpelling().str())) {
          break;
        }
      }
      // continue to parse the expression.
      paranBalanced--;
      prevToken = currTok;
      continue;
    }

    // Process all the operators.
    if (isOperator(getTokenSpelling().str())) {

      // // Handle unary minus.
      // if (currTok == Token::minus &&
      //     (isOperator(prevToken) || prevToken == Token::l_paren ||
      //      prevToken == Token::unknown)) {
      //   currTok = Token::unary_minus;
      // }

      if (prevToken.getKind() == Token::l_paren) {
        emitError(getLoc(), "Error in expression");
        return nullptr;
      }

      auto OpInfo = OperatorInfo(
          state.context->getBinaryOperator(getTokenSpelling().str()));
      auto currPrecedence = getOpPrecedence(OpInfo);
      while (!opsStack.empty()) {

        auto stackPrecedence = getOpPrecedence(opsStack.top());
        if (stackPrecedence < currPrecedence) {
          break;
        }
        pushOperation(valueStack, opsStack, loc);
      }

      // Push the current token to ops.
      // Handle unary minus.
      opsStack.push(OpInfo);
      consumeToken(); // op
      prevToken = currTok;
      continue;
    }
    // Any other operator, just break;
    break;
  }

  // Build all the parsed expressions.
  while (!opsStack.empty()) {
    pushOperation(valueStack, opsStack, loc);
  }

  assert(valueStack.size() == 1);
  assert(opsStack.empty());
  return valueStack.top();
}