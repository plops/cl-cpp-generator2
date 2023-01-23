#include <clang-c/Index.h>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  if (!(2 == argc)) {
    std::cerr << "usage: " << argv[0] << " <source file>" << std::endl;
    return 1;
  }
  auto cmdline = "-I/usr/lib/llvm-11/include/";
  auto cmdlineArgs = std::vector<const char *>();
  auto index = clang_createIndex(0, 0);
  cmdlineArgs.push_back(cmdline);
  auto tu = clang_parseTranslationUnit(index, argv[1], cmdlineArgs.data(), 1,
                                       nullptr, 0, CXTranslationUnit_None);
  if (!(tu)) {
    std::cerr << "failed to parse " << argv[1] << std::endl;
  }
  auto code = clang_getTranslationUnitSpelling(tu);
  std::cout << clang_getCString(code) << std::endl;
  clang_visitChildren(
      clang_getTranslationUnitCursor(tu),
      [](CXCursor c, CXCursor parent, CXClientData data) {
        if (CXCursor_ParenExpr == clang_getCursorKind(c)) {
          auto child = clang_getChild(c);
          if (clang_isExpression(clang_getCursorKind(child))) {
            clang_replaceChildRange(parent, c, child,
                                    clang_getNextSibling(child), data);
          }
        }
        return CXChildVisit_Recurse;
      },
      nullptr);
  clang_disposeString(code);
  clang_disposeTranslationUnit(tu);
  clang_disposeIndex(index);
  return 0;
}
