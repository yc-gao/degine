func.func @addi_same_index(%a : index) -> index {
    %c0 = arith.constant 0 : index
    %x = arith.addi %a, %c0 : index

    %y = arith.addi %a, %a : index

    %z = arith.addi %x, %y : index
    return %z : index
}
