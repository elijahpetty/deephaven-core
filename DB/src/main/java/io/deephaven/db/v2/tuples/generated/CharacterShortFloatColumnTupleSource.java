package io.deephaven.db.v2.tuples.generated;

import io.deephaven.datastructures.util.SmartKey;
import io.deephaven.db.util.tuples.generated.CharShortFloatTuple;
import io.deephaven.db.v2.sources.ColumnSource;
import io.deephaven.db.v2.sources.WritableSource;
import io.deephaven.db.v2.sources.chunk.Attributes;
import io.deephaven.db.v2.sources.chunk.CharChunk;
import io.deephaven.db.v2.sources.chunk.Chunk;
import io.deephaven.db.v2.sources.chunk.FloatChunk;
import io.deephaven.db.v2.sources.chunk.ObjectChunk;
import io.deephaven.db.v2.sources.chunk.ShortChunk;
import io.deephaven.db.v2.sources.chunk.WritableChunk;
import io.deephaven.db.v2.sources.chunk.WritableObjectChunk;
import io.deephaven.db.v2.tuples.AbstractTupleSource;
import io.deephaven.db.v2.tuples.ThreeColumnTupleSourceFactory;
import io.deephaven.db.v2.tuples.TupleSource;
import io.deephaven.util.type.TypeUtils;
import org.jetbrains.annotations.NotNull;


/**
 * <p>{@link TupleSource} that produces key column values from {@link ColumnSource} types Character, Short, and Float.
 * <p>Generated by {@link io.deephaven.db.v2.tuples.TupleSourceCodeGenerator}.
 */
@SuppressWarnings({"unused", "WeakerAccess"})
public class CharacterShortFloatColumnTupleSource extends AbstractTupleSource<CharShortFloatTuple> {

    /** {@link ThreeColumnTupleSourceFactory} instance to create instances of {@link CharacterShortFloatColumnTupleSource}. **/
    public static final ThreeColumnTupleSourceFactory<CharShortFloatTuple, Character, Short, Float> FACTORY = new Factory();

    private final ColumnSource<Character> columnSource1;
    private final ColumnSource<Short> columnSource2;
    private final ColumnSource<Float> columnSource3;

    public CharacterShortFloatColumnTupleSource(
            @NotNull final ColumnSource<Character> columnSource1,
            @NotNull final ColumnSource<Short> columnSource2,
            @NotNull final ColumnSource<Float> columnSource3
    ) {
        super(columnSource1, columnSource2, columnSource3);
        this.columnSource1 = columnSource1;
        this.columnSource2 = columnSource2;
        this.columnSource3 = columnSource3;
    }

    @Override
    public final CharShortFloatTuple createTuple(final long indexKey) {
        return new CharShortFloatTuple(
                columnSource1.getChar(indexKey),
                columnSource2.getShort(indexKey),
                columnSource3.getFloat(indexKey)
        );
    }

    @Override
    public final CharShortFloatTuple createPreviousTuple(final long indexKey) {
        return new CharShortFloatTuple(
                columnSource1.getPrevChar(indexKey),
                columnSource2.getPrevShort(indexKey),
                columnSource3.getPrevFloat(indexKey)
        );
    }

    @Override
    public final CharShortFloatTuple createTupleFromValues(@NotNull final Object... values) {
        return new CharShortFloatTuple(
                TypeUtils.unbox((Character)values[0]),
                TypeUtils.unbox((Short)values[1]),
                TypeUtils.unbox((Float)values[2])
        );
    }

    @Override
    public final CharShortFloatTuple createTupleFromReinterpretedValues(@NotNull final Object... values) {
        return new CharShortFloatTuple(
                TypeUtils.unbox((Character)values[0]),
                TypeUtils.unbox((Short)values[1]),
                TypeUtils.unbox((Float)values[2])
        );
    }

    @SuppressWarnings("unchecked")
    @Override
    public final <ELEMENT_TYPE> void exportElement(@NotNull final CharShortFloatTuple tuple, final int elementIndex, @NotNull final WritableSource<ELEMENT_TYPE> writableSource, final long destinationIndexKey) {
        if (elementIndex == 0) {
            writableSource.set(destinationIndexKey, tuple.getFirstElement());
            return;
        }
        if (elementIndex == 1) {
            writableSource.set(destinationIndexKey, tuple.getSecondElement());
            return;
        }
        if (elementIndex == 2) {
            writableSource.set(destinationIndexKey, tuple.getThirdElement());
            return;
        }
        throw new IndexOutOfBoundsException("Invalid element index " + elementIndex + " for export");
    }

    @Override
    public final Object exportToExternalKey(@NotNull final CharShortFloatTuple tuple) {
        return new SmartKey(
                TypeUtils.box(tuple.getFirstElement()),
                TypeUtils.box(tuple.getSecondElement()),
                TypeUtils.box(tuple.getThirdElement())
        );
    }

    @Override
    public final Object exportElement(@NotNull final CharShortFloatTuple tuple, int elementIndex) {
        if (elementIndex == 0) {
            return TypeUtils.box(tuple.getFirstElement());
        }
        if (elementIndex == 1) {
            return TypeUtils.box(tuple.getSecondElement());
        }
        if (elementIndex == 2) {
            return TypeUtils.box(tuple.getThirdElement());
        }
        throw new IllegalArgumentException("Bad elementIndex for 3 element tuple: " + elementIndex);
    }

    @Override
    public final Object exportElementReinterpreted(@NotNull final CharShortFloatTuple tuple, int elementIndex) {
        if (elementIndex == 0) {
            return TypeUtils.box(tuple.getFirstElement());
        }
        if (elementIndex == 1) {
            return TypeUtils.box(tuple.getSecondElement());
        }
        if (elementIndex == 2) {
            return TypeUtils.box(tuple.getThirdElement());
        }
        throw new IllegalArgumentException("Bad elementIndex for 3 element tuple: " + elementIndex);
    }

    @Override
    public Class<CharShortFloatTuple> getNativeType() {
        return CharShortFloatTuple.class;
    }

    @Override
    protected void convertChunks(@NotNull WritableChunk<? super Attributes.Values> destination, int chunkSize, Chunk<Attributes.Values> [] chunks) {
        WritableObjectChunk<CharShortFloatTuple, ? super Attributes.Values> destinationObjectChunk = destination.asWritableObjectChunk();
        CharChunk<Attributes.Values> chunk1 = chunks[0].asCharChunk();
        ShortChunk<Attributes.Values> chunk2 = chunks[1].asShortChunk();
        FloatChunk<Attributes.Values> chunk3 = chunks[2].asFloatChunk();
        for (int ii = 0; ii < chunkSize; ++ii) {
            destinationObjectChunk.set(ii, new CharShortFloatTuple(chunk1.get(ii), chunk2.get(ii), chunk3.get(ii)));
        }
        destinationObjectChunk.setSize(chunkSize);
    }

    /** {@link ThreeColumnTupleSourceFactory} for instances of {@link CharacterShortFloatColumnTupleSource}. **/
    private static final class Factory implements ThreeColumnTupleSourceFactory<CharShortFloatTuple, Character, Short, Float> {

        private Factory() {
        }

        @Override
        public TupleSource<CharShortFloatTuple> create(
                @NotNull final ColumnSource<Character> columnSource1,
                @NotNull final ColumnSource<Short> columnSource2,
                @NotNull final ColumnSource<Float> columnSource3
        ) {
            return new CharacterShortFloatColumnTupleSource(
                    columnSource1,
                    columnSource2,
                    columnSource3
            );
        }
    }
}
