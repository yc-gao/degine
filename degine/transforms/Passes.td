include "mlir/Pass/PassBase.td"

def AsyncFromDataFlowPass: Pass<"async-from-dataflow", "func::FuncOp"> {
    let dependentDialects = [
        "func::FuncDialect",
        "async::AsyncDialect"
    ];
}

