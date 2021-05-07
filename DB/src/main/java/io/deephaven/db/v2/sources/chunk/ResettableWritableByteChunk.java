/* ---------------------------------------------------------------------------------------------------------------------
 * AUTO-GENERATED CLASS - DO NOT EDIT MANUALLY - for any changes edit ResettableWritableCharChunk and regenerate
 * ------------------------------------------------------------------------------------------------------------------ */
package io.deephaven.db.v2.sources.chunk;
import io.deephaven.db.tables.utils.ArrayUtils;
import io.deephaven.db.v2.sources.chunk.Attributes.Any;
import io.deephaven.db.v2.sources.chunk.util.pools.MultiChunkPool;
import io.deephaven.db.v2.utils.ChunkUtils;

/**
 * {@link ResettableWritableChunk} implementation for byte data.
 *
 * @IncludeAll
 */
public final class ResettableWritableByteChunk<ATTR_BASE extends Any> extends WritableByteChunk implements ResettableWritableChunk<ATTR_BASE> {

    public static <ATTR_BASE extends Any> ResettableWritableByteChunk<ATTR_BASE> makeResettableChunk() {
        return MultiChunkPool.forThisThread().getByteChunkPool().takeResettableWritableByteChunk();
    }

    public static <ATTR_BASE extends Any> ResettableWritableByteChunk<ATTR_BASE> makeResettableChunkForPool() {
        return new ResettableWritableByteChunk<>();
    }

    private ResettableWritableByteChunk(byte[] data, int offset, int capacity) {
        super(data, offset, capacity);
    }

    private ResettableWritableByteChunk() {
        this(ArrayUtils.EMPTY_BYTE_ARRAY, 0, 0);
    }

    @Override
    public final ResettableWritableByteChunk slice(int offset, int capacity) {
        ChunkUtils.checkSliceArgs(size, offset, capacity);
        return new ResettableWritableByteChunk<>(data, this.offset + offset, capacity);
    }

    @Override
    public final <ATTR extends ATTR_BASE> WritableByteChunk<ATTR> resetFromChunk(WritableChunk<ATTR> other, int offset, int capacity) {
        return resetFromTypedChunk(other.asWritableByteChunk(), offset, capacity);
    }

    @Override
    public final <ATTR extends ATTR_BASE> WritableByteChunk<ATTR> resetFromArray(Object array, int offset, int capacity) {
        final byte[] typedArray = (byte[])array;
        return resetFromTypedArray(typedArray, offset, capacity);
    }

    public final <ATTR extends ATTR_BASE> WritableByteChunk<ATTR> resetFromArray(Object array) {
        final byte[] typedArray = (byte[])array;
        return resetFromTypedArray(typedArray, 0, typedArray.length);
    }

    @Override
    public final <ATTR extends ATTR_BASE> WritableByteChunk<ATTR> clear() {
        return resetFromArray(ArrayUtils.EMPTY_BYTE_ARRAY, 0, 0);
    }

    public final <ATTR extends ATTR_BASE> WritableByteChunk<ATTR> resetFromTypedChunk(WritableByteChunk<ATTR> other, int offset, int capacity) {
        ChunkUtils.checkSliceArgs(other.size, offset, capacity);
        return resetFromTypedArray(other.data, other.offset + offset, capacity);
    }

    public final <ATTR extends ATTR_BASE> WritableByteChunk<ATTR> resetFromTypedArray(byte[] data, int offset, int capacity) {
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
        MultiChunkPool.forThisThread().getByteChunkPool().giveResettableWritableByteChunk(this);
    }
}
