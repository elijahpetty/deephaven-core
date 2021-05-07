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
import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * <p>3-Tuple (triple) key class composed of double, double, and Object elements.
 * <p>Generated by {@link io.deephaven.db.util.tuples.TupleCodeGenerator}.
 */
public class DoubleDoubleObjectTuple implements Comparable<DoubleDoubleObjectTuple>, Externalizable, StreamingExternalizable, CanonicalizableTuple<DoubleDoubleObjectTuple> {

    private static final long serialVersionUID = 1L;

    private double element1;
    private double element2;
    private Object element3;

    private transient int cachedHashCode;

    public DoubleDoubleObjectTuple(
            final double element1,
            final double element2,
            final Object element3
    ) {
        initialize(
                element1,
                element2,
                element3
        );
    }

    /** Public no-arg constructor for {@link Externalizable} support only. <em>Application code should not use this!</em> **/
    public DoubleDoubleObjectTuple() {
    }

    private void initialize(
            final double element1,
            final double element2,
            final Object element3
    ) {
        this.element1 = element1;
        this.element2 = element2;
        this.element3 = element3;
        cachedHashCode = ((31 +
                Double.hashCode(element1)) * 31 +
                Double.hashCode(element2)) * 31 +
                Objects.hashCode(element3);
    }

    public final double getFirstElement() {
        return element1;
    }

    public final double getSecondElement() {
        return element2;
    }

    public final Object getThirdElement() {
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
        final DoubleDoubleObjectTuple typedOther = (DoubleDoubleObjectTuple) other;
        // @formatter:off
        return element1 == typedOther.element1 &&
               element2 == typedOther.element2 &&
               Objects.equals(element3, typedOther.element3);
        // @formatter:on
    }

    @Override
    public final int compareTo(@NotNull final DoubleDoubleObjectTuple other) {
        if (this == other) {
            return 0;
        }
        int comparison;
        // @formatter:off
        return 0 != (comparison = DBLanguageFunctionUtil.compareTo(element1, other.element1)) ? comparison :
               0 != (comparison = DBLanguageFunctionUtil.compareTo(element2, other.element2)) ? comparison :
               DBLanguageFunctionUtil.compareTo((Comparable)element3, (Comparable)other.element3);
        // @formatter:on
    }

    @Override
    public void writeExternal(@NotNull final ObjectOutput out) throws IOException {
        out.writeDouble(element1);
        out.writeDouble(element2);
        out.writeObject(element3);
    }

    @Override
    public void readExternal(@NotNull final ObjectInput in) throws IOException, ClassNotFoundException {
        initialize(
                in.readDouble(),
                in.readDouble(),
                in.readObject()
        );
    }

    @Override
    public void writeExternalStreaming(@NotNull final ObjectOutput out, @NotNull final TIntObjectMap<SerializationUtils.Writer> cachedWriters) throws IOException {
        out.writeDouble(element1);
        out.writeDouble(element2);
        StreamingExternalizable.writeObjectElement(out, cachedWriters, 2, element3);
    }

    @Override
    public void readExternalStreaming(@NotNull final ObjectInput in, @NotNull final TIntObjectMap<SerializationUtils.Reader> cachedReaders) throws Exception {
        initialize(
                in.readDouble(),
                in.readDouble(),
                StreamingExternalizable.readObjectElement(in, cachedReaders, 2)
        );
    }

    @Override
    public String toString() {
        return "DoubleDoubleObjectTuple{" +
                element1 + ", " +
                element2 + ", " +
                element3 + '}';
    }

    @Override
    public DoubleDoubleObjectTuple canonicalize(@NotNull final UnaryOperator<Object> canonicalizer) {
        final Object canonicalizedElement3 = canonicalizer.apply(element3);
        return canonicalizedElement3 == element3
                ? this : new DoubleDoubleObjectTuple(element1, element2, canonicalizedElement3);
    }
}
