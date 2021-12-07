/* ---------------------------------------------------------------------------------------------------------------------
 * AUTO-GENERATED CLASS - DO NOT EDIT MANUALLY - for any changes edit CharDupExpandKernel and regenerate
 * ------------------------------------------------------------------------------------------------------------------ */
package io.deephaven.engine.table.impl.join.dupexpand;

import io.deephaven.chunk.*;
import io.deephaven.chunk.attributes.Any;
import io.deephaven.chunk.attributes.ChunkLengths;

public class DoubleDupExpandKernel implements DupExpandKernel {
    public static final DoubleDupExpandKernel INSTANCE = new DoubleDupExpandKernel();

    private DoubleDupExpandKernel() {} // use through the instance

    @Override
    public void expandDuplicates(int expandedSize, WritableChunk<? extends Any> chunkToExpand, IntChunk<ChunkLengths> keyRunLengths) {
        expandDuplicates(expandedSize, chunkToExpand.asWritableDoubleChunk(), keyRunLengths);
    }

    public static void expandDuplicates(int expandedSize, WritableDoubleChunk<? extends Any> chunkToExpand, IntChunk<ChunkLengths> keyRunLengths) {
        if (expandedSize == 0) {
            return;
        }

        int wpos = expandedSize;
        int rpos = chunkToExpand.size() - 1;
        chunkToExpand.setSize(expandedSize);

        for (; rpos >= 0; --rpos) {
            final int len = keyRunLengths.get(rpos);
            chunkToExpand.fillWithValue(wpos - len, len, chunkToExpand.get(rpos));
            wpos -= len;
        }
    }
}