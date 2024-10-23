func.func @addi_same_index(%a : index) -> index {
    %c0 = arith.constant 0 : index
    %y = arith.addi %a, %c0 : index

    %token0, %x = async.execute -> !async.value<index> {
        %x = arith.addi %a, %c0 : index
        async.yield %x : index
    }

    %token2, %z = async.execute
                    [%token0]
                    (%x as %unwrapped_x : !async.value<index>)
                    -> !async.value<index> {
        %z = arith.addi %unwrapped_x, %a : index
        async.yield %z : index
    }

    return %y : index
}

