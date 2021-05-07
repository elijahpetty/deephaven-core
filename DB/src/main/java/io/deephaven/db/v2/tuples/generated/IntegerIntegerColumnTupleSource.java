package io.deephaven.db.v2.tuples.generated;

import io.deephaven.datastructures.util.SmartKey;
import io.deephaven.db.util.tuples.generated.IntIntTuple;
import io.deephaven.db.v2.sources.ColumnSource;
import io.deephaven.db.v2.sources.WritableSource;
import io.deephaven.db.v2.sources.chunk.Attributes;
import io.deephaven.db.v2.sources.chunk.Chunk;
import io.deephaven.db.v2.sources.chunk.IntChunk;
import io.deephaven.db.v2.sources.chunk.ObjectChunk;
import io.deephaven.db.v2.sources.chunk.WritableChunk;
import io.deephaven.db.v2.sources.chunk.WritableObjectChunk;
import io.deephaven.db.v2.tuples.AbstractTupleSource;
import io.deephaven.db.v2.tuples.TupleSource;
import io.deephaven.db.v2.tuples.TwoColumnTupleSourceFactory;
import io.deephaven.util.type.TypeUtils;
import org.jetbrains.annotations.NotNull;


/**
 * <p>{@link TupleSource} that produces key column values from {@link ColumnSource} types Integer and Integer.
 * <p>Generated by {@link io.deephaven.db.v2.tuples.TupleSourceCodeGenerator}.
 */
@SuppressWarnings({"unused", "WeakerAccess"})
public class IntegerIntegerColumnTupleSource extends AbstractTupleSource<IntIntTuple> {

    /** {@link TwoColumnTupleSourceFactory} instance to create instances of {@link IntegerIntegerColumnTupleSource}. **/
    public static final TwoColumnTupleSourceFactory<IntIntTuple, Integer, Integer> FACTORY = new Factory();

    private final ColumnSource<Integer> columnSource1;
    private final ColumnSource<Integer> columnSource2;

    public IntegerIntegerColumnTupleSource(
            @NotNull final ColumnSource<Integer> columnSource1,
            @NotNull final ColumnSource<Integer> columnSource2
    ) {
        super(columnSource1, columnSource2);
        this.columnSource1 = columnSource1;
        this.columnSource2 = columnSource2;
    }

    @Override
    public final IntIntTuple createTuple(final long indexKey) {
        return new IntIntTuple(
                columnSource1.getInt(indexKey),
                columnSource2.getInt(indexKey)
        );
    }

    @Override
    public final IntIntTuple createPreviousTuple(final long indexKey) {
        return new IntIntTuple(
                columnSource1.getPrevInt(indexKey),
                columnSource2.getPrevInt(indexKey)
        );
    }

    @Override
    public final IntIntTuple createTupleFromValues(@NotNull final Object... values) {
        return new IntIntTuple(
                TypeUtils.unbox((Integer)values[0]),
                TypeUtils.unbox((Integer)values[1])
        );
    }

    @Override
    public final IntIntTuple createTupleFromReinterpretedValues(@NotNull final Object... values) {
        return new IntIntTuple(
                TypeUtils.unbox((Integer)values[0]),
                TypeUtils.unbox((Integer)values[1])
        );
    }

    @SuppressWarnings("unchecked")
    @Override
    public final <ELEMENT_TYPE> void exportElement(@NotNull final IntIntTuple tuple, final int elementIndex, @NotNull final WritableSource<ELEMENT_TYPE> writableSource, final long destinationIndexKey) {
        if (elementIndex == 0) {
            writableSource.set(destinationIndexKey, tuple.getFirstElement());
            return;
        }
        if (elementIndex == 1) {
            writableSource.set(destinationIndexKey, tuple.getSecondElement());
            return;
        }
        throw new IndexOutOfBoundsException("Invalid element index " + elementIndex + " for export");
    }

    @Override
    public final Object exportToExternalKey(@NotNull final IntIntTuple tuple) {
        return new SmartKey(
                TypeUtils.box(tuple.getFirstElement()),
                TypeUtils.box(tuple.getSecondElement())
        );
    }

    @Override
    public final Object exportElement(@NotNull final IntIntTuple tuple, int elementIndex) {
        if (elementIndex == 0) {
            return TypeUtils.box(tuple.getFirstElement());
        }
        if (elementIndex == 1) {
            return TypeUtils.box(tuple.getSecondElement());
        }
        throw new IllegalArgumentException("Bad elementIndex for 2 element tuple: " + elementIndex);
    }

    @Override
    public final Object exportElementReinterpreted(@NotNull final IntIntTuple tuple, int elementIndex) {
        if (elementIndex == 0) {
            return TypeUtils.box(tuple.getFirstElement());
        }
        if (elementIndex == 1) {
            return TypeUtils.box(tuple.getSecondElement());
        }
        throw new IllegalArgumentException("Bad elementIndex for 2 element tuple: " + elementIndex);
    }

    @Override
    public Class<IntIntTuple> getNativeType() { return IntIntTuple.class; }

    protected void convertChunks(@NotNull WritableChunk<? super Attributes.Values> destination, int chunkSize, Chunk<Attributes.Values> [] chunks) {
        WritableObjectChunk<IntIntTuple, ? super Attributes.Values> destinationObjectChunk = destination.asWritableObjectChunk();
        IntChunk<Attributes.Values> chunk1 = chunks[0].asIntChunk();
        IntChunk<Attributes.Values> chunk2 = chunks[1].asIntChunk();
        for (int ii = 0; ii < chunkSize; ++ii) {
            destinationObjectChunk.set(ii, new IntIntTuple(chunk1.get(ii), chunk2.get(ii)));
        }
        destination.setSize(chunkSize);
    }

    /** {@link TwoColumnTupleSourceFactory} for instances of {@link IntegerIntegerColumnTupleSource}. **/
    private static final class Factory implements TwoColumnTupleSourceFactory<IntIntTuple, Integer, Integer> {

        private Factory() {
        }

        @Override
        public TupleSource<IntIntTuple> create(
                @NotNull final ColumnSource<Integer> columnSource1,
                @NotNull final ColumnSource<Integer> columnSource2
        ) {
            return new IntegerIntegerColumnTupleSource(
                    columnSource1,
                    columnSource2
            );
        }
    }
}
