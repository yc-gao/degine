#include "llvm/Support/CommandLine.h"

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "degine optimizer");
  return 0;
}
