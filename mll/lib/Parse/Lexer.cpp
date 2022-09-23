//===- Lexer.cpp - MLL Lexer Implementation ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lexer for the MLIR textual form.
//
//===----------------------------------------------------------------------===//

#include "mll/Parse/Lexer.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SourceMgr.h"

using namespace mll;

/*
// Returns true if 'c' is an allowable punctuation character: [$._-]
// Returns false otherwise.
static bool isPunct(char c) {
  return c == '$' || c == '.' || c == '_' || c == '-';
}
*/

Lexer::Lexer(const llvm::SourceMgr &sourceMgr, MLLContext *context)
    : sourceMgr(sourceMgr), context(context) {
  auto bufferID = sourceMgr.getMainFileID();
  curBuffer = sourceMgr.getMemoryBuffer(bufferID)->getBuffer();
  curPtr = curBuffer.begin();
}

/// Encode the specified source location information into an attribute for
/// attachment to the IR.
Location Lexer::getEncodedSourceLocation(SMLoc loc) {
  auto &sourceMgr = getSourceMgr();
  unsigned mainFileID = sourceMgr.getMainFileID();

  // TODO: Fix performance issues in SourceMgr::getLineAndColumn so that we can
  //       use it here.
  auto &bufferInfo = sourceMgr.getBufferInfo(mainFileID);
  unsigned lineNo = bufferInfo.getLineNumber(loc.getPointer());
  unsigned column =
      (loc.getPointer() - bufferInfo.getPointerForLineNumber(lineNo)) + 1;
  auto *buffer = sourceMgr.getMemoryBuffer(mainFileID);

  return Location::get(context, buffer->getBufferIdentifier(), lineNo, column);
}

/// emitError - Emit an error message and return an Token::error token.
Token Lexer::emitError(const char *loc, const llvm::Twine &message) {
  llvm::errs() << "\n Error in lexing the tokens\n : " << message;
  return formToken(Token::error, loc);
}

Token Lexer::lexToken() {
  while (true) {
    const char *tokStart = curPtr;

    // Lex the next token.
    switch (*curPtr++) {
    default:
      // Handle bare identifiers.
      if (isalpha(curPtr[-1]))
        return lexBareIdentifierOrKeyword(tokStart);

      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");

    case ' ':
    case '\t':
    case '\n':
    case '\r':
      // Handle whitespace.
      continue;

    case '_':
      // Handle bare identifiers.
      return lexBareIdentifierOrKeyword(tokStart);

    case 0:
      // This may either be a nul character in the source file or may be the EOF
      // marker that llvm::MemoryBuffer guarantees will be there.
      if (curPtr - 1 == curBuffer.end())
        return formToken(Token::eof, tokStart);
      continue;

    case ':':
      return formToken(Token::colon, tokStart);
    case ',':
      return formToken(Token::comma, tokStart);
    case '.':
      return formToken(Token::dot, tokStart);
    case '(':
      return formToken(Token::l_paren, tokStart);
    case ')':
      return formToken(Token::r_paren, tokStart);
    case '{':
      return formToken(Token::l_brace, tokStart);
    case '}':
      return formToken(Token::r_brace, tokStart);
    case '[':
      return formToken(Token::l_square, tokStart);
    case ']':
      return formToken(Token::r_square, tokStart);
    case '<':
      if (*curPtr == '=') {
        ++curPtr;
        return formToken(Token::less_equal, tokStart);
      }
      return formToken(Token::less, tokStart);
    case '>':
      if (*curPtr == '=') {
        ++curPtr;
        return formToken(Token::greater_equal, tokStart);
      }
      return formToken(Token::greater, tokStart);
    case '=':
      if (*curPtr == '=') {
        ++curPtr;
        return formToken(Token::compare_equal, tokStart);
      }
      return formToken(Token::equal, tokStart);

    case '+':
      return formToken(Token::plus, tokStart);
    case '*':
      return formToken(Token::star, tokStart);
    case '-':
      if (*curPtr == '>') {
        ++curPtr;
        return formToken(Token::arrow, tokStart);
      }
      if (isdigit(*curPtr)) {
        return lexNumber(curPtr);
      }
      return formToken(Token::minus, tokStart);

    case '?':
      return formToken(Token::question, tokStart);

    case '|':
      return formToken(Token::vertical_bar, tokStart);

    case '/':
      if (*curPtr == '/') {
        skipComment();
        continue;
      }
      return formToken(Token::forward_slash, tokStart);

    case '!':
      if (*curPtr == '=') {
        ++curPtr;
        return formToken(Token::not_equal, tokStart);
      }
      LLVM_FALLTHROUGH;
    case '^':
    case '"':
      return lexString(tokStart);

    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      return lexNumber(tokStart);
    }
  }
}

/// Lex a bare identifier or keyword that starts with a letter.
///
///   bare-id ::= (letter|[_]) (letter|digit|[_])*
///   integer-type ::= `[su]?i[1-9][0-9]*`
///
Token Lexer::lexBareIdentifierOrKeyword(const char *tokStart) {
  // Match the rest of the identifier regex: [0-9a-zA-Z_$]*
  while (isalpha(*curPtr) || isdigit(*curPtr) || *curPtr == '_')
    ++curPtr;

  // Check to see if this identifier is a keyword.
  StringRef spelling(tokStart, curPtr - tokStart);

  Token::Kind kind = llvm::StringSwitch<Token::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) .Case(#SPELLING, Token::kw_##SPELLING)
#include "mll/Parse/TokenKinds.def"
                         .Default(Token::identifier);

  return Token(kind, spelling);
}

/// Skip a comment line, starting with a '//'.
///
///   TODO: add a regex for comments here and to the spec.
///
void Lexer::skipComment() {
  // Advance over the second '/' in a '//' comment.
  assert(*curPtr == '/');
  ++curPtr;

  while (true) {
    switch (*curPtr++) {
    case '\n':
    case '\r':
      // Newline is end of comment.
      return;
    case 0:
      // If this is the end of the buffer, end the comment.
      if (curPtr - 1 == curBuffer.end()) {
        --curPtr;
        return;
      }
      LLVM_FALLTHROUGH;
    default:
      // Skip over other characters.
      break;
    }
  }
}

/// Lex a number literal.
///
///   integer-literal ::= [-]?digit+ | `0x` hex_digit+
///   float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
///
Token Lexer::lexNumber(const char *tokStart) {
  assert(isdigit(curPtr[-1]) || curPtr[-1] == '-');

  bool hasNegativeSign = curPtr[-1] == '-';

  // Handle the hexadecimal case.
  if (curPtr[-1] == '0' && *curPtr == 'x') {
    // If we see stuff like 0xi32, this is a literal `0` followed by an
    // identifier `xi32`, stop after `0`.
    if (!isxdigit(curPtr[1]))
      return formToken(Token::integer, tokStart);

    curPtr += 2;
    while (isxdigit(*curPtr))
      ++curPtr;

    return formToken(Token::integer, tokStart);
  }

  // Handle the normal decimal case.
  while (isdigit(*curPtr))
    ++curPtr;

  if (hasNegativeSign)
    tokStart--;

  if (*curPtr != '.')
    return formToken(Token::integer, tokStart);
  ++curPtr;

  // Skip over [0-9]*([eE][-+]?[0-9]+)?
  while (isdigit(*curPtr))
    ++curPtr;

  if (*curPtr == 'e' || *curPtr == 'E') {
    if (isdigit(static_cast<unsigned char>(curPtr[1])) ||
        ((curPtr[1] == '-' || curPtr[1] == '+') &&
         isdigit(static_cast<unsigned char>(curPtr[2])))) {
      curPtr += 2;
      while (isdigit(*curPtr))
        ++curPtr;
    }
  }
  return formToken(Token::floatliteral, tokStart);
}

/// Lex an identifier that starts with a prefix followed by suffix-id.
///
///   attribute-id  ::= `#` suffix-id
///   ssa-id        ::= '%' suffix-id
///   block-id      ::= '^' suffix-id
///   type-id       ::= '!' suffix-id
///   suffix-id     ::= digit+ | (letter|id-punct) (letter|id-punct|digit)*
///   id-punct      ::= `$` | `.` | `_` | `-`
///

/// Lex a string literal.
///
///   string-literal ::= '"' [^"\n\f\v\r]* '"'
///
/// TODO: define escaping rules.
Token Lexer::lexString(const char *tokStart) {
  assert(curPtr[-1] == '"');

  while (true) {
    switch (*curPtr++) {
    case '"':
      return formToken(Token::string, tokStart);
    case 0:
      // If this is a random nul character in the middle of a string, just
      // include it.  If it is the end of file, then it is an error.
      if (curPtr - 1 != curBuffer.end())
        continue;
      LLVM_FALLTHROUGH;
    case '\n':
    case '\v':
    case '\f':
      return emitError(curPtr - 1, "expected '\"' in string literal");
    case '\\':
      // Handle explicitly a few escapes.
      if (*curPtr == '"' || *curPtr == '\\' || *curPtr == 'n' || *curPtr == 't')
        ++curPtr;
      else if (llvm::isHexDigit(*curPtr) && llvm::isHexDigit(curPtr[1]))
        // Support \xx for two hex digits.
        curPtr += 2;
      else
        return emitError(curPtr - 1, "unknown escape in string literal");
      continue;

    default:
      continue;
    }
  }
}
