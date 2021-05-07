package io.deephaven.db.v2.tuples.generated;

import io.deephaven.datastructures.util.SmartKey;
import io.deephaven.db.util.BooleanUtils;
import io.deephaven.db.util.tuples.generated.ByteObjectCharTuple;
import io.deephaven.db.v2.sources.ColumnSource;
import io.deephaven.db.v2.sources.WritableSource;
import io.deephaven.db.v2.sources.chunk.Attributes;
import io.deephaven.db.v2.sources.chunk.CharChunk;
import io.deephaven.db.v2.sources.chunk.Chunk;
import io.deephaven.db.v2.sources.chunk.ObjectChunk;
import io.deephaven.db.v2.sources.chunk.WritableChunk;
import io.deephaven.db.v2.sources.chunk.WritableObjectChunk;
import io.deephaven.db.v2.tuples.AbstractTupleSource;
import io.deephaven.db.v2.tuples.ThreeColumnTupleSourceFactory;
import io.deephaven.db.v2.tuples.TupleSource;
import io.deephaven.util.type.TypeUtils;
import org.jetbrains.annotations.NotNull;


/**
 * <p>{@link TupleSource} that produces key column values from {@link ColumnSource} types Boolean, Object, and Character.
 * <p>Generated by {@link io.deephaven.db.v2.tuples.TupleSourceCodeGenerator}.
 */
@SuppressWarnings({"unused", "WeakerAccess"})
public class BooleanObjectCharacterColumnTupleSource extends AbstractTupleSource<ByteObjectCharTuple> {

    /** {@link ThreeColumnTupleSourceFactory} instance to create instances of {@link BooleanObjectCharacterColumnTupleSource}. **/
    public static final ThreeColumnTupleSourceFactory<ByteObjectCharTuple, Boolean, Object, Character> FACTORY = new Factory();

    private final ColumnSource<Boolean> columnSource1;
    private final ColumnSource<Object> columnSource2;
    private final ColumnSource<Character> columnSource3;

    public BooleanObjectCharacterColumnTupleSource(
            @NotNull final ColumnSource<Boolean> columnSource1,
            @NotNull final ColumnSource<Object> columnSource2,
            @NotNull final ColumnSource<Character> columnSource3
    ) {
        super(columnSource1, columnSource2, columnSource3);
        this.columnSource1 = columnSource1;
        this.columnSource2 = columnSource2;
        this.columnSource3 = columnSource3;
    }

    @Override
    public final ByteObjectCharTuple createTuple(final long indexKey) {
        return new ByteObjectCharTuple(
                BooleanUtils.booleanAsByte(columnSource1.getBoolean(indexKey)),
                columnSource2.get(indexKey),
                columnSource3.getChar(indexKey)
        );
    }

    @Override
    public final ByteObjectCharTuple createPreviousTuple(final long indexKey) {
        return new ByteObjectCharTuple(
                BooleanUtils.booleanAsByte(columnSource1.getPrevBoolean(indexKey)),
                columnSource2.getPrev(indexKey),
                columnSource3.getPrevChar(indexKey)
        );
    }

    @Override
    public final ByteObjectCharTuple createTupleFromValues(@NotNull final Object... values) {
        return new ByteObjectCharTuple(
                BooleanUtils.booleanAsByte((Boolean)values[0]),
                values[1],
                TypeUtils.unbox((Character)values[2])
        );
    }

    @Override
    public final ByteObjectCharTuple createTupleFromReinterpretedValues(@NotNull final Object... values) {
        return new ByteObjectCharTuple(
                BooleanUtils.booleanAsByte((Boolean)values[0]),
                values[1],
                TypeUtils.unbox((Character)values[2])
        );
    }

    @SuppressWarnings("unchecked")
    @Override
    public final <ELEMENT_TYPE> void exportElement(@NotNull final ByteObjectCharTuple tuple, final int elementIndex, @NotNull final WritableSource<ELEMENT_TYPE> writableSource, final long destinationIndexKey) {
        if (elementIndex == 0) {
            writableSource.set(destinationIndexKey, (ELEMENT_TYPE) BooleanUtils.byteAsBoolean(tuple.getFirstElement()));
            return;
        }
        if (elementIndex == 1) {
            writableSource.set(destinationIndexKey, (ELEMENT_TYPE) tuple.getSecondElement());
            return;
        }
        if (elementIndex == 2) {
            writableSource.set(destinationIndexKey, tuple.getThirdElement());
            return;
        }
        throw new IndexOutOfBoundsException("Invalid element index " + elementIndex + " for export");
    }

    @Override
    public final Object exportToExternalKey(@NotNull final ByteObjectCharTuple tuple) {
        return new SmartKey(
                BooleanUtils.byteAsBoolean(tuple.getFirstElement()),
                tuple.getSecondElement(),
                TypeUtils.box(tuple.getThirdElement())
        );
    }

    @Override
    public final Object exportElement(@NotNull final ByteObjectCharTuple tuple, int elementIndex) {
        if (elementIndex == 0) {
            return BooleanUtils.byteAsBoolean(tuple.getFirstElement());
        }
        if (elementIndex == 1) {
            return tuple.getSecondElement();
        }
        if (elementIndex == 2) {
            return TypeUtils.box(tuple.getThirdElement());
        }
        throw new IllegalArgumentException("Bad elementIndex for 3 element tuple: " + elementIndex);
    }

    @Override
    public final Object exportElementReinterpreted(@NotNull final ByteObjectCharTuple tuple, int elementIndex) {
        if (elementIndex == 0) {
            return BooleanUtils.byteAsBoolean(tuple.getFirstElement());
        }
        if (elementIndex == 1) {
            return tuple.getSecondElement();
        }
        if (elementIndex == 2) {
            return TypeUtils.box(tuple.getThirdElement());
        }
        throw new IllegalArgumentException("Bad elementIndex for 3 element tuple: " + elementIndex);
    }

    @Override
    public Class<ByteObjectCharTuple> getNativeType() {
        return ByteObjectCharTuple.class;
    }

    @Override
    protected void convertChunks(@NotNull WritableChunk<? super Attributes.Values> destination, int chunkSize, Chunk<Attributes.Values> [] chunks) {
        WritableObjectChunk<ByteObjectCharTuple, ? super Attributes.Values> destinationObjectChunk = destination.asWritableObjectChunk();
        ObjectChunk<Boolean, Attributes.Values> chunk1 = chunks[0].asObjectChunk();
        ObjectChunk<Object, Attributes.Values> chunk2 = chunks[1].asObjectChunk();
        CharChunk<Attributes.Values> chunk3 = chunks[2].asCharChunk();
        for (int ii = 0; ii < chunkSize; ++ii) {
            destinationObjectChunk.set(ii, new ByteObjectCharTuple(BooleanUtils.booleanAsByte(chunk1.get(ii)), chunk2.get(ii), chunk3.get(ii)));
        }
        destinationObjectChunk.setSize(chunkSize);
    }

    /** {@link ThreeColumnTupleSourceFactory} for instances of {@link BooleanObjectCharacterColumnTupleSource}. **/
    private static final class Factory implements ThreeColumnTupleSourceFactory<ByteObjectCharTuple, Boolean, Object, Character> {

        private Factory() {
        }

        @Override
        public TupleSource<ByteObjectCharTuple> create(
                @NotNull final ColumnSource<Boolean> columnSource1,
                @NotNull final ColumnSource<Object> columnSource2,
                @NotNull final ColumnSource<Character> columnSource3
        ) {
            return new BooleanObjectCharacterColumnTupleSource(
                    columnSource1,
                    columnSource2,
                    columnSource3
            );
        }
    }
}
