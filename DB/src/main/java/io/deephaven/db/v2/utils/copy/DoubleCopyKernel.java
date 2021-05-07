/* ---------------------------------------------------------------------------------------------------------------------
 * AUTO-GENERATED CLASS - DO NOT EDIT MANUALLY - for any changes edit CharCopyKernel and regenerate
 * ------------------------------------------------------------------------------------------------------------------ */
package io.deephaven.db.v2.utils.copy;

import io.deephaven.db.v2.sources.chunk.*;

import static io.deephaven.db.v2.sources.chunk.Attributes.Any;

public class DoubleCopyKernel implements CopyKernel {
    public static final DoubleCopyKernel INSTANCE = new DoubleCopyKernel();

    private DoubleCopyKernel() {} // use through the instance

    /**
     * Copy values from baseInput / overInput into output.
     * <p>
     * Pseudo-implementation: output[i] = useOverInput.forBit(i) ? overInput[i] : baseInput[i];
     * <p>
     * Note that useOverInput should cover the same data range as baseInput such that
     * {@code useOverInput.length == (overInput.length + 63) / 64} is true.
     *
     * @param output       the output chunk
     * @param baseInput    the input array to use when bit in useOverInput is zero (array)
     * @param overInput    the input array to use when bit in useOverInput is one (array)
     * @param useOverInput the bitset array to indicate whether to use baseInput or overInput for each element
     * @param srcOffset    the offset in baseInput/overInput
     * @param dstOffset    the offset in output
     * @param length       the number of elements to copy
     */
    public static <T extends Any> void conditionalCopy(
            WritableDoubleChunk<T> output, double[] baseInput, double[] overInput, long[] useOverInput,
            int srcOffset, int dstOffset, int length)
    {
        int bitsSet = 0;
        final int bitsetLen = (length + 63) >> 6;
        final int bitsetOffset = srcOffset >> 6;
        for (int i = 0; i < bitsetLen; ++i) {
            bitsSet += Long.bitCount(useOverInput[i + bitsetOffset]);
        }
        final int totalBits = bitsetLen << 6;
        final boolean flipBase = bitsSet > totalBits / 2;

        // mem-copy from baseline
        output.copyFromTypedArray(flipBase ? overInput : baseInput, srcOffset, dstOffset, length);

        final int srcEndOffset = srcOffset + length;
        for (int ii = CopyKernel.Utils.nextSetBit(useOverInput, srcOffset, srcEndOffset, flipBase);
             ii < srcEndOffset;
             ii = CopyKernel.Utils.nextSetBit(useOverInput, ii + 1, srcEndOffset, flipBase)) {
            output.set(dstOffset + ii - srcOffset, flipBase ? baseInput[ii] : overInput[ii]);
        }
    }

    @Override
    public <T extends Any> void conditionalCopy(
            WritableChunk<T> output, Object baseInput, Object overInput, long[] useOverInput,
            int srcOffset, int dstOffset, int length)
    {
        double[] typedBaseInput = (double[]) baseInput;
        double[] typedOverInput = (double[]) overInput;
        conditionalCopy(output.asWritableDoubleChunk(), typedBaseInput, typedOverInput, useOverInput, srcOffset, dstOffset, length);
    }
}
