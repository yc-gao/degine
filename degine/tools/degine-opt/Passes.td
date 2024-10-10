include "mlir/Pass/PassBase.td"

def DegineDemoPass : Pass<"degine-demo", "func::FuncOp"> {
  let summary = "Canonicalizes dynamic StableHLO ops into static ops.";
}

