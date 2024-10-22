func.func @addi_same_index_basic(%a : index) -> index {
    %x = arith.addi %a, %a : index
    return %x : index
}

