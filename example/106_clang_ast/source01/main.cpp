#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <iostream>
class ASTPrinter : public clang::ASTConsumer {
public:
  virtual bool HandleTopLevelDecl(clang::DeclGroupRef dg) {
    for (auto &d : dg) {
      d->dump();
    }
    return true;
  }
};
class ASTAction : public clang::ASTFrontendAction {
public:
  virtual std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &ci, llvm::StringRef file) {
    return std::unique_ptr<clang::ASTConsumer>(new ASTPrinter());
  }
};

int main(int argc, char **argv) {}
