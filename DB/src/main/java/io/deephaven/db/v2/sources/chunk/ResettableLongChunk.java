/* ---------------------------------------------------------------------------------------------------------------------
 * AUTO-GENERATED CLASS - DO NOT EDIT MANUALLY - for any changes edit ResettableCharChunk and regenerate
 * ------------------------------------------------------------------------------------------------------------------ */
package io.deephaven.db.v2.sources.chunk;
import io.deephaven.db.tables.utils.ArrayUtils;
import io.deephaven.db.v2.sources.chunk.Attributes.Any;
import io.deephaven.db.v2.sources.chunk.util.pools.MultiChunkPool;
import io.deephaven.db.v2.utils.ChunkUtils;

/**
 * {@link ResettableReadOnlyChunk} implementation for long data.
 *
 * @IncludeAll
 */
public final class ResettableLongChunk<ATTR_UPPER extends Any> extends LongChunk implements ResettableReadOnlyChunk<ATTR_UPPER> {

    public static <ATTR_BASE extends Any> ResettableLongChunk<ATTR_BASE> makeResettableChunk() {
        return MultiChunkPool.forThisThread().getLongChunkPool().takeResettableLongChunk();
    }

    public static <ATTR_BASE extends Any> ResettableLongChunk<ATTR_BASE> makeResettableChunkForPool() {
        return new ResettableLongChunk<>();
    }

    private ResettableLongChunk(long[] data, int offset, int capacity) {
        super(data, offset, capacity);
    }

    private ResettableLongChunk() {
        this(ArrayUtils.EMPTY_LONG_ARRAY, 0, 0);
    }

    @Override
    public final ResettableLongChunk slice(int offset, int capacity) {
        ChunkUtils.checkSliceArgs(size, offset, capacity);
        return new ResettableLongChunk(data, this.offset + offset, capacity);
    }

    @Override
    public final <ATTR extends ATTR_UPPER> LongChunk<ATTR> resetFromChunk(Chunk<? extends ATTR> other, int offset, int capacity) {
        return resetFromTypedChunk(other.asLongChunk(), offset, capacity);
    }

    @Override
    public final <ATTR extends ATTR_UPPER> LongChunk<ATTR> resetFromArray(Object array, int offset, int capacity) {
        final long[] typedArray = (long[])array;
        return resetFromTypedArray(typedArray, offset, capacity);
    }

    @Override
    public final <ATTR extends ATTR_UPPER> LongChunk<ATTR> resetFromArray(Object array) {
        final long[] typedArray = (long[])array;
        return resetFromTypedArray(typedArray, 0, typedArray.length);
    }

    @Override
    public final <ATTR extends ATTR_UPPER> LongChunk<ATTR> clear() {
        return resetFromArray(ArrayUtils.EMPTY_LONG_ARRAY, 0, 0);
    }

    public final <ATTR extends ATTR_UPPER> LongChunk<ATTR> resetFromTypedChunk(LongChunk<? extends ATTR> other, int offset, int capacity) {
        ChunkUtils.checkSliceArgs(other.size, offset, capacity);
        return resetFromTypedArray(other.data, other.offset + offset, capacity);
    }

    public final <ATTR extends ATTR_UPPER> LongChunk<ATTR> resetFromTypedArray(long[] data, int offset, int capacity) {
        ChunkUtils.checkArrayArgs(data.length, offset, capacity);
        this.data = data;
        this.offset = offset;
        this.capacity = capacity;
        this.size = capacity;
        //noinspection unchecked
        return this;
    }

    @Override
    public final void close() {
        MultiChunkPool.forThisThread().getLongChunkPool().giveResettableLongChunk(this);
    }
}
