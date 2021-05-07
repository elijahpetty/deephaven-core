package io.deephaven.db.util.tuples.generated;

import io.deephaven.db.tables.lang.DBLanguageFunctionUtil;
import io.deephaven.db.util.serialization.SerializationUtils;
import io.deephaven.db.util.serialization.StreamingExternalizable;
import io.deephaven.db.util.tuples.CanonicalizableTuple;
import gnu.trove.map.TIntObjectMap;
import org.jetbrains.annotations.NotNull;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.function.UnaryOperator;

/**
 * <p>3-Tuple (triple) key class composed of float, float, and short elements.
 * <p>Generated by {@link io.deephaven.db.util.tuples.TupleCodeGenerator}.
 */
public class FloatFloatShortTuple implements Comparable<FloatFloatShortTuple>, Externalizable, StreamingExternalizable, CanonicalizableTuple<FloatFloatShortTuple> {

    private static final long serialVersionUID = 1L;

    private float element1;
    private float element2;
    private short element3;

    private transient int cachedHashCode;

    public FloatFloatShortTuple(
            final float element1,
            final float element2,
            final short element3
    ) {
        initialize(
                element1,
                element2,
                element3
        );
    }

    /** Public no-arg constructor for {@link Externalizable} support only. <em>Application code should not use this!</em> **/
    public FloatFloatShortTuple() {
    }

    private void initialize(
            final float element1,
            final float element2,
            final short element3
    ) {
        this.element1 = element1;
        this.element2 = element2;
        this.element3 = element3;
        cachedHashCode = ((31 +
                Float.hashCode(element1)) * 31 +
                Float.hashCode(element2)) * 31 +
                Short.hashCode(element3);
    }

    public final float getFirstElement() {
        return element1;
    }

    public final float getSecondElement() {
        return element2;
    }

    public final short getThirdElement() {
        return element3;
    }

    @Override
    public final int hashCode() {
        return cachedHashCode;
    }

    @Override
    public final boolean equals(final Object other) {
        if (this == other) {
            return true;
        }
        if (other == null || getClass() != other.getClass()) {
            return false;
        }
        final FloatFloatShortTuple typedOther = (FloatFloatShortTuple) other;
        // @formatter:off
        return element1 == typedOther.element1 &&
               element2 == typedOther.element2 &&
               element3 == typedOther.element3;
        // @formatter:on
    }

    @Override
    public final int compareTo(@NotNull final FloatFloatShortTuple other) {
        if (this == other) {
            return 0;
        }
        int comparison;
        // @formatter:off
        return 0 != (comparison = DBLanguageFunctionUtil.compareTo(element1, other.element1)) ? comparison :
               0 != (comparison = DBLanguageFunctionUtil.compareTo(element2, other.element2)) ? comparison :
               DBLanguageFunctionUtil.compareTo(element3, other.element3);
        // @formatter:on
    }

    @Override
    public void writeExternal(@NotNull final ObjectOutput out) throws IOException {
        out.writeFloat(element1);
        out.writeFloat(element2);
        out.writeShort(element3);
    }

    @Override
    public void readExternal(@NotNull final ObjectInput in) throws IOException, ClassNotFoundException {
        initialize(
                in.readFloat(),
                in.readFloat(),
                in.readShort()
        );
    }

    @Override
    public void writeExternalStreaming(@NotNull final ObjectOutput out, @NotNull final TIntObjectMap<SerializationUtils.Writer> cachedWriters) throws IOException {
        out.writeFloat(element1);
        out.writeFloat(element2);
        out.writeShort(element3);
    }

    @Override
    public void readExternalStreaming(@NotNull final ObjectInput in, @NotNull final TIntObjectMap<SerializationUtils.Reader> cachedReaders) throws Exception {
        initialize(
                in.readFloat(),
                in.readFloat(),
                in.readShort()
        );
    }

    @Override
    public String toString() {
        return "FloatFloatShortTuple{" +
                element1 + ", " +
                element2 + ", " +
                element3 + '}';
    }

    @Override
    public FloatFloatShortTuple canonicalize(@NotNull final UnaryOperator<Object> canonicalizer) {
        return this;
    }
}
