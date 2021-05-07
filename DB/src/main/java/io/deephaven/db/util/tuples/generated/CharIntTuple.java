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
 * <p>2-Tuple (double) key class composed of char and int elements.
 * <p>Generated by {@link io.deephaven.db.util.tuples.TupleCodeGenerator}.
 */
public class CharIntTuple implements Comparable<CharIntTuple>, Externalizable, StreamingExternalizable, CanonicalizableTuple<CharIntTuple> {

    private static final long serialVersionUID = 1L;

    private char element1;
    private int element2;

    private transient int cachedHashCode;

    public CharIntTuple(
            final char element1,
            final int element2
    ) {
        initialize(
                element1,
                element2
        );
    }

    /** Public no-arg constructor for {@link Externalizable} support only. <em>Application code should not use this!</em> **/
    public CharIntTuple() {
    }

    private void initialize(
            final char element1,
            final int element2
    ) {
        this.element1 = element1;
        this.element2 = element2;
        cachedHashCode = (31 +
                Character.hashCode(element1)) * 31 +
                Integer.hashCode(element2);
    }

    public final char getFirstElement() {
        return element1;
    }

    public final int getSecondElement() {
        return element2;
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
        final CharIntTuple typedOther = (CharIntTuple) other;
        // @formatter:off
        return element1 == typedOther.element1 &&
               element2 == typedOther.element2;
        // @formatter:on
    }

    @Override
    public final int compareTo(@NotNull final CharIntTuple other) {
        if (this == other) {
            return 0;
        }
        int comparison;
        // @formatter:off
        return 0 != (comparison = DBLanguageFunctionUtil.compareTo(element1, other.element1)) ? comparison :
               DBLanguageFunctionUtil.compareTo(element2, other.element2);
        // @formatter:on
    }

    @Override
    public void writeExternal(@NotNull final ObjectOutput out) throws IOException {
        out.writeChar(element1);
        out.writeInt(element2);
    }

    @Override
    public void readExternal(@NotNull final ObjectInput in) throws IOException, ClassNotFoundException {
        initialize(
                in.readChar(),
                in.readInt()
        );
    }

    @Override
    public void writeExternalStreaming(@NotNull final ObjectOutput out, @NotNull final TIntObjectMap<SerializationUtils.Writer> cachedWriters) throws IOException {
        out.writeChar(element1);
        out.writeInt(element2);
    }

    @Override
    public void readExternalStreaming(@NotNull final ObjectInput in, @NotNull final TIntObjectMap<SerializationUtils.Reader> cachedReaders) throws Exception {
        initialize(
                in.readChar(),
                in.readInt()
        );
    }

    @Override
    public String toString() {
        return "CharIntTuple{" +
                element1 + ", " +
                element2 + '}';
    }

    @Override
    public CharIntTuple canonicalize(@NotNull final UnaryOperator<Object> canonicalizer) {
        return this;
    }
}
