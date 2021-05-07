/* ---------------------------------------------------------------------------------------------------------------------
 * AUTO-GENERATED CLASS - DO NOT EDIT MANUALLY - for any changes edit FloatChunkedAvgOperator and regenerate
 * ------------------------------------------------------------------------------------------------------------------ */
/*
 * Copyright (c) 2016-2021 Deephaven Data Labs and Patent Pending
 */

package io.deephaven.db.v2.by;

import io.deephaven.db.v2.sources.ColumnSource;
import io.deephaven.db.v2.sources.DoubleArraySource;
import io.deephaven.db.v2.sources.chunk.*;
import io.deephaven.db.v2.sources.chunk.Attributes.*;
import org.apache.commons.lang3.mutable.MutableInt;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import static io.deephaven.db.util.NullSafeAddition.plusDouble;
import static io.deephaven.db.v2.by.ComboAggregateFactory.*;

class DoubleChunkedAvgOperator extends FpChunkedNonNormalCounter implements IterativeChunkedAggregationOperator {
    private final String name;
    private final boolean exposeInternalColumns;
    private final DoubleArraySource resultColumn = new DoubleArraySource();
    private final DoubleArraySource runningSum = new DoubleArraySource();
    private final NonNullCounter nonNullCounter = new NonNullCounter();

    DoubleChunkedAvgOperator(String name, boolean exposeInternalColumns) {
        this.name = name;
        this.exposeInternalColumns = exposeInternalColumns;
    }

    @Override
    public void addChunk(BucketedContext context, Chunk<? extends Values> values, LongChunk<? extends KeyIndices> inputIndices, IntChunk<KeyIndices> destinations, IntChunk<ChunkPositions> startPositions, IntChunk<ChunkLengths> length, WritableBooleanChunk<Values> stateModified) {
        final DoubleChunk<? extends Values> asDoubleChunk = values.asDoubleChunk();
        for (int ii = 0; ii < startPositions.size(); ++ii) {
            final int startPosition = startPositions.get(ii);
            final long destination = destinations.get(startPosition);
            stateModified.set(ii, addChunk(asDoubleChunk, destination, startPosition, length.get(ii)));
        }
    }

    @Override
    public void removeChunk(BucketedContext context, Chunk<? extends Values> values, LongChunk<? extends KeyIndices> inputIndices, IntChunk<KeyIndices> destinations, IntChunk<ChunkPositions> startPositions, IntChunk<ChunkLengths> length, WritableBooleanChunk<Values> stateModified) {
        final DoubleChunk<? extends Values> asDoubleChunk = values.asDoubleChunk();
        for (int ii = 0; ii < startPositions.size(); ++ii) {
            final int startPosition = startPositions.get(ii);
            final long destination = destinations.get(startPosition);
            stateModified.set(ii, removeChunk(asDoubleChunk, destination, startPosition, length.get(ii)));
        }
    }

    @Override
    public boolean addChunk(SingletonContext context, int chunkSize, Chunk<? extends Values> values, LongChunk<? extends KeyIndices> inputIndices, long destination) {
        return addChunk(values.asDoubleChunk(), destination, 0, values.size());
    }

    @Override
    public boolean removeChunk(SingletonContext context, int chunkSize, Chunk<? extends Values> values, LongChunk<? extends KeyIndices> inputIndices, long destination) {
        return removeChunk(values.asDoubleChunk(), destination, 0, values.size());
    }

    private boolean addChunk(DoubleChunk<? extends Values> values, long destination, int chunkStart, int chunkSize) {
        final MutableInt chunkNormalCount = new MutableInt(0);
        final MutableInt chunkNanCount = new MutableInt(0);
        final MutableInt chunkInfinityCount = new MutableInt(0);
        final MutableInt chunkMinusInfinityCount = new MutableInt(0);

        final double sum = SumDoubleChunk.sumDoubleChunk(values, chunkStart, chunkSize, chunkNormalCount, chunkNanCount, chunkInfinityCount, chunkMinusInfinityCount);

        final long totalNormal = nonNullCounter.addNonNullUnsafe(destination, chunkNormalCount.intValue());
        final long totalNanCount = updateNanCount(destination, chunkNanCount.intValue());
        final long totalPositiveInfinityCount = updatePositiveInfinityCount(destination, chunkInfinityCount.intValue());
        final long totalNegativeInfinityCount = updateNegativeInfinityCount(destination, chunkMinusInfinityCount.intValue());

        if (chunkNormalCount.intValue() > 0) {
            final double newSum = plusDouble(runningSum.getUnsafe(destination), sum);
            runningSum.set(destination, newSum);
            updateResultWithNewSum(destination, totalNormal, totalNanCount, totalPositiveInfinityCount, totalNegativeInfinityCount, newSum);
        } else {
            updateResultSumUnchanged(destination, totalNormal, totalNanCount, totalPositiveInfinityCount, totalNegativeInfinityCount);
        }
        return true;
    }

    private boolean removeChunk(DoubleChunk<? extends Values> values, long destination, int chunkStart, int chunkSize) {
        final MutableInt chunkNormalCount = new MutableInt(0);
        final MutableInt chunkNanCount = new MutableInt(0);
        final MutableInt chunkInfinityCount = new MutableInt(0);
        final MutableInt chunkMinusInfinityCount = new MutableInt(0);

        final double sum = SumDoubleChunk.sumDoubleChunk(values, chunkStart, chunkSize, chunkNormalCount, chunkNanCount, chunkInfinityCount, chunkMinusInfinityCount);
        if (chunkNormalCount.intValue() == 0 && chunkNanCount.intValue() == 0 && chunkInfinityCount.intValue() == 0 && chunkMinusInfinityCount.intValue() == 0) {
            return false;
        }

        final long totalNormal = nonNullCounter.addNonNull(destination, -chunkNormalCount.intValue());
        final long totalNanCount = updateNanCount(destination, -chunkNanCount.intValue());
        final long totalPositiveInfinityCount = updatePositiveInfinityCount(destination, -chunkInfinityCount.intValue());
        final long totalNegativeInfinityCount = updateNegativeInfinityCount(destination, -chunkMinusInfinityCount.intValue());

        final double newSum;
        if (chunkNormalCount.intValue() > 0) {
            newSum = plusDouble(runningSum.getUnsafe(destination), -sum);
            runningSum.set(destination, newSum);
            updateResultWithNewSum(destination, totalNormal, totalNanCount, totalPositiveInfinityCount, totalNegativeInfinityCount, newSum);
        } else {
            updateResultSumUnchanged(destination, totalNormal, totalNanCount, totalPositiveInfinityCount, totalNegativeInfinityCount);
        }
        return true;
    }

    private void updateResultWithNewSum(long destination, long totalNormal, long totalNanCount, long totalInfinityCount, long totalNegativeInfinityCount, double newSum) {
        if (totalNanCount > 0 || (totalInfinityCount > 0 && totalNegativeInfinityCount > 0)) {
            resultColumn.set(destination, Double.NaN);
        } else if (totalInfinityCount > 0) {
            resultColumn.set(destination, Double.POSITIVE_INFINITY);
        } else if (totalNegativeInfinityCount > 0) {
            resultColumn.set(destination, Double.NEGATIVE_INFINITY);
        } else {
            resultColumn.set(destination, newSum / totalNormal);
        }
    }

    private void updateResultSumUnchanged(long destination, long totalNormal, long totalNanCount, long totalInfinityCount, long totalNegativeInfinityCount) {
        if (totalNanCount > 0 || totalNormal == 0 || (totalInfinityCount > 0 && totalNegativeInfinityCount > 0)) {
            resultColumn.set(destination, Double.NaN);
        } else if (totalInfinityCount > 0) {
            resultColumn.set(destination, Double.POSITIVE_INFINITY);
        } else if (totalNegativeInfinityCount > 0) {
            resultColumn.set(destination, Double.NEGATIVE_INFINITY);
        } else {
            resultColumn.set(destination, runningSum.getUnsafe(destination) / totalNormal);
        }
    }

    @Override
    public void ensureCapacity(long tableSize) {
        resultColumn.ensureCapacity(tableSize);
        nonNullCounter.ensureCapacity(tableSize);
        runningSum.ensureCapacity(tableSize);
        ensureNonNormalCapacity(tableSize);
    }

    @Override
    public Map<String, ? extends ColumnSource<?>> getResultColumns() {
        if (exposeInternalColumns) {
            final Map<String, ColumnSource<?>> results = new LinkedHashMap<>();
            results.put(name, resultColumn);
            results.put(name + ROLLUP_RUNNING_SUM_COLUMN_ID + ROLLUP_COLUMN_SUFFIX, runningSum);
            results.put(name + ROLLUP_NONNULL_COUNT_COLUMN_ID + ROLLUP_COLUMN_SUFFIX, nonNullCounter.getColumnSource());
            results.putAll(fpInternalColumnSources(name));
            return results;
        } else {
            return Collections.singletonMap(name, resultColumn);
        }
    }

    @Override
    public void startTrackingPrevValues() {
        resultColumn.startTrackingPrevValues();
        if (exposeInternalColumns) {
            runningSum.startTrackingPrevValues();
            nonNullCounter.startTrackingPrevValues();
            startTrackingPrevFpCounterValues();
        }
    }
}