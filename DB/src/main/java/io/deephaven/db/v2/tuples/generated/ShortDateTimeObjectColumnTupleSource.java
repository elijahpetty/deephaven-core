package io.deephaven.db.v2.tuples.generated;

import io.deephaven.datastructures.util.SmartKey;
import io.deephaven.db.tables.utils.DBDateTime;
import io.deephaven.db.tables.utils.DBTimeUtils;
import io.deephaven.db.util.tuples.generated.ShortLongObjectTuple;
import io.deephaven.db.v2.sources.ColumnSource;
import io.deephaven.db.v2.sources.WritableSource;
import io.deephaven.db.v2.sources.chunk.Attributes;
import io.deephaven.db.v2.sources.chunk.Chunk;
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
 * <p>{@link TupleSource} that produces key column values from {@link ColumnSource} types Short, DBDateTime, and Object.
 * <p>Generated by {@link io.deephaven.db.v2.tuples.TupleSourceCodeGenerator}.
 */
@SuppressWarnings({"unused", "WeakerAccess"})
public class ShortDateTimeObjectColumnTupleSource extends AbstractTupleSource<ShortLongObjectTuple> {

    /** {@link ThreeColumnTupleSourceFactory} instance to create instances of {@link ShortDateTimeObjectColumnTupleSource}. **/
    public static final ThreeColumnTupleSourceFactory<ShortLongObjectTuple, Short, DBDateTime, Object> FACTORY = new Factory();

    private final ColumnSource<Short> columnSource1;
    private final ColumnSource<DBDateTime> columnSource2;
    private final ColumnSource<Object> columnSource3;

    public ShortDateTimeObjectColumnTupleSource(
            @NotNull final ColumnSource<Short> columnSource1,
            @NotNull final ColumnSource<DBDateTime> columnSource2,
            @NotNull final ColumnSource<Object> columnSource3
    ) {
        super(columnSource1, columnSource2, columnSource3);
        this.columnSource1 = columnSource1;
        this.columnSource2 = columnSource2;
        this.columnSource3 = columnSource3;
    }

    @Override
    public final ShortLongObjectTuple createTuple(final long indexKey) {
        return new ShortLongObjectTuple(
                columnSource1.getShort(indexKey),
                DBTimeUtils.nanos(columnSource2.get(indexKey)),
                columnSource3.get(indexKey)
        );
    }

    @Override
    public final ShortLongObjectTuple createPreviousTuple(final long indexKey) {
        return new ShortLongObjectTuple(
                columnSource1.getPrevShort(indexKey),
                DBTimeUtils.nanos(columnSource2.getPrev(indexKey)),
                columnSource3.getPrev(indexKey)
        );
    }

    @Override
    public final ShortLongObjectTuple createTupleFromValues(@NotNull final Object... values) {
        return new ShortLongObjectTuple(
                TypeUtils.unbox((Short)values[0]),
                DBTimeUtils.nanos((DBDateTime)values[1]),
                values[2]
        );
    }

    @Override
    public final ShortLongObjectTuple createTupleFromReinterpretedValues(@NotNull final Object... values) {
        return new ShortLongObjectTuple(
                TypeUtils.unbox((Short)values[0]),
                DBTimeUtils.nanos((DBDateTime)values[1]),
                values[2]
        );
    }

    @SuppressWarnings("unchecked")
    @Override
    public final <ELEMENT_TYPE> void exportElement(@NotNull final ShortLongObjectTuple tuple, final int elementIndex, @NotNull final WritableSource<ELEMENT_TYPE> writableSource, final long destinationIndexKey) {
        if (elementIndex == 0) {
            writableSource.set(destinationIndexKey, tuple.getFirstElement());
            return;
        }
        if (elementIndex == 1) {
            writableSource.set(destinationIndexKey, (ELEMENT_TYPE) DBTimeUtils.nanosToTime(tuple.getSecondElement()));
            return;
        }
        if (elementIndex == 2) {
            writableSource.set(destinationIndexKey, (ELEMENT_TYPE) tuple.getThirdElement());
            return;
        }
        throw new IndexOutOfBoundsException("Invalid element index " + elementIndex + " for export");
    }

    @Override
    public final Object exportToExternalKey(@NotNull final ShortLongObjectTuple tuple) {
        return new SmartKey(
                TypeUtils.box(tuple.getFirstElement()),
                DBTimeUtils.nanosToTime(tuple.getSecondElement()),
                tuple.getThirdElement()
        );
    }

    @Override
    public final Object exportElement(@NotNull final ShortLongObjectTuple tuple, int elementIndex) {
        if (elementIndex == 0) {
            return TypeUtils.box(tuple.getFirstElement());
        }
        if (elementIndex == 1) {
            return DBTimeUtils.nanosToTime(tuple.getSecondElement());
        }
        if (elementIndex == 2) {
            return tuple.getThirdElement();
        }
        throw new IllegalArgumentException("Bad elementIndex for 3 element tuple: " + elementIndex);
    }

    @Override
    public final Object exportElementReinterpreted(@NotNull final ShortLongObjectTuple tuple, int elementIndex) {
        if (elementIndex == 0) {
            return TypeUtils.box(tuple.getFirstElement());
        }
        if (elementIndex == 1) {
            return DBTimeUtils.nanosToTime(tuple.getSecondElement());
        }
        if (elementIndex == 2) {
            return tuple.getThirdElement();
        }
        throw new IllegalArgumentException("Bad elementIndex for 3 element tuple: " + elementIndex);
    }

    @Override
    public Class<ShortLongObjectTuple> getNativeType() {
        return ShortLongObjectTuple.class;
    }

    @Override
    protected void convertChunks(@NotNull WritableChunk<? super Attributes.Values> destination, int chunkSize, Chunk<Attributes.Values> [] chunks) {
        WritableObjectChunk<ShortLongObjectTuple, ? super Attributes.Values> destinationObjectChunk = destination.asWritableObjectChunk();
        ShortChunk<Attributes.Values> chunk1 = chunks[0].asShortChunk();
        ObjectChunk<DBDateTime, Attributes.Values> chunk2 = chunks[1].asObjectChunk();
        ObjectChunk<Object, Attributes.Values> chunk3 = chunks[2].asObjectChunk();
        for (int ii = 0; ii < chunkSize; ++ii) {
            destinationObjectChunk.set(ii, new ShortLongObjectTuple(chunk1.get(ii), DBTimeUtils.nanos(chunk2.get(ii)), chunk3.get(ii)));
        }
        destinationObjectChunk.setSize(chunkSize);
    }

    /** {@link ThreeColumnTupleSourceFactory} for instances of {@link ShortDateTimeObjectColumnTupleSource}. **/
    private static final class Factory implements ThreeColumnTupleSourceFactory<ShortLongObjectTuple, Short, DBDateTime, Object> {

        private Factory() {
        }

        @Override
        public TupleSource<ShortLongObjectTuple> create(
                @NotNull final ColumnSource<Short> columnSource1,
                @NotNull final ColumnSource<DBDateTime> columnSource2,
                @NotNull final ColumnSource<Object> columnSource3
        ) {
            return new ShortDateTimeObjectColumnTupleSource(
                    columnSource1,
                    columnSource2,
                    columnSource3
            );
        }
    }
}
