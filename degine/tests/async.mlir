func.func @addi_same_index(%a : index) -> index {
    %c0 = arith.constant 0 : index
    %x = arith.addi %a, %c0 : index

    %y = arith.addi %a, %a : index

    %z = arith.addi %x, %y : index
    return %z : index
}

func.func @addi_same_index_async(%a : index) -> index {
    %token0, %x = async.execute -> !async.value<index> {
        %c0 = arith.constant 0 : index
        %x = arith.addi %a, %c0 : index
        async.yield %x : index
    }

    %token1, %y = async.execute -> !async.value<index> {
        %y = arith.addi %a, %a : index
        async.yield %y : index
    }

    %token2, %z = async.execute
                    [%token0, %token1]
                    (%x as %unwrapped_x : !async.value<index>, %y as %unwrapped_y : !async.value<index>)
                    -> !async.value<index> {
        %z = arith.addi %unwrapped_x, %unwrapped_y : index
        async.yield %z : index
    }
    %unwrapped_z = async.await %z: !async.value<index>
    return %unwrapped_z : index
}
