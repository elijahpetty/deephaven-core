/*
 * Copyright (c) 2016-2021 Deephaven Data Labs and Patent Pending
 */

package io.deephaven.db.plot.datasets.multiseries;

import io.deephaven.db.plot.AxesImpl;
import io.deephaven.db.plot.datasets.DynamicSeriesNamer;
import io.deephaven.db.plot.datasets.xy.XYDataSeriesInternal;
import io.deephaven.db.plot.datasets.xy.XYDataSeriesTableArray;
import io.deephaven.db.plot.filters.SelectableDataSetSwappableTable;
import io.deephaven.db.plot.util.ArgumentValidations;
import io.deephaven.db.plot.util.tables.SwappableTable;
import io.deephaven.db.plot.util.tables.TableHandle;
import io.deephaven.db.v2.BaseTable;


/**
 * A {@link AbstractMultiSeries} collection that holds and generates {@link XYDataSeriesInternal}.
 */
public class MultiXYSeriesSwappable extends AbstractSwappableMultiSeries<XYDataSeriesInternal> {

    private static final long serialVersionUID = 1274883777622079921L;

    private final String xCol;
    private final String yCol;

    /**
     * Creates a MultiXYSeriesSwappable instance.
     *
     * @param axes axes on which this {@link MultiSeries} will be plotted
     * @param id data series id
     * @param name               series name
     * @param swappableTable swappable table
     * @param xCol column in {@code t} that holds the x-variable data
     * @param yCol column in {@code t} that holds the y-variable data
     * @param byColumns column(s) in {@code t} that holds the grouping data
     */
    public MultiXYSeriesSwappable(final AxesImpl axes, final int id, final Comparable name, final SwappableTable swappableTable, final String xCol, final String yCol, final String[] byColumns) {
        super(axes, id, name, swappableTable, xCol, yCol, byColumns);
        ArgumentValidations.assertIsNumericOrTime(swappableTable.getTableDefinition(), xCol, getPlotInfo());
        ArgumentValidations.assertIsNumericOrTime(swappableTable.getTableDefinition(), yCol, getPlotInfo());
        this.xCol = xCol;
        this.yCol = yCol;
    }

    /**
     * Creates a copy of a series using a different Axes.
     *
     * @param series series to copy.
     * @param axes new axes to use.
     */
    private MultiXYSeriesSwappable(final MultiXYSeriesSwappable series, final AxesImpl axes) {
        super(series, axes);
        this.xCol = series.xCol;
        this.yCol = series.yCol;
    }

    @Override
    public XYDataSeriesInternal createSeries(String seriesName, final BaseTable t, final DynamicSeriesNamer seriesNamer){
        seriesName = makeSeriesName(seriesName, seriesNamer);

        final TableHandle h = new TableHandle(t, xCol, yCol);
        addTableHandle(h);

        return new XYDataSeriesTableArray(axes(), -1, seriesName, h, xCol, yCol);
    }

    ////////////////////////////// CODE BELOW HERE IS GENERATED -- DO NOT EDIT BY HAND //////////////////////////////
    ////////////////////////////// TO REGENERATE RUN GenerateMultiSeries //////////////////////////////
    ////////////////////////////// AND THEN RUN GenerateFigureImmutable //////////////////////////////

    @Override public void initializeSeries(XYDataSeriesInternal series) {
        $$initializeSeries$$(series);
    }

    @Override public <T extends io.deephaven.gui.color.Paint> MultiXYSeriesSwappable pointColorByY(final groovy.lang.Closure<T> colors, final Object... keys) {
        return pointColorByY(new io.deephaven.db.plot.util.functions.ClosureFunction<>(colors), keys);
    }



    @Override public <T extends io.deephaven.gui.color.Paint> MultiXYSeriesSwappable pointColorByY(final java.util.function.Function<java.lang.Double, T> colors, final Object... keys) {
        final String newColumn = io.deephaven.db.plot.datasets.ColumnNameConstants.POINT_COLOR + this.hashCode();
        applyFunction(colors, newColumn, getY(), io.deephaven.gui.color.Paint.class);
        chart().figure().registerFigureFunction(new io.deephaven.db.plot.util.functions.FigureImplFunction(f -> f.pointColor(new SelectableDataSetSwappableTable(getSwappableTable()), newColumn, keys), this));
        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> errorBarColorSeriesNameToStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> errorBarColorSeriesNameToStringMap() {
        return errorBarColorSeriesNameToStringMap;
    }
    @Override public MultiXYSeriesSwappable errorBarColor(final java.lang.String color, final Object... keys) {
        if(keys == null || keys.length == 0) {
            errorBarColorSeriesNameToStringMap.setDefault(color);
        } else {
            errorBarColorSeriesNameToStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                color);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Integer> errorBarColorSeriesNameTointMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Integer> errorBarColorSeriesNameTointMap() {
        return errorBarColorSeriesNameTointMap;
    }
    @Override public MultiXYSeriesSwappable errorBarColor(final int color, final Object... keys) {
        if(keys == null || keys.length == 0) {
            errorBarColorSeriesNameTointMap.setDefault(color);
        } else {
            errorBarColorSeriesNameTointMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                color);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.gui.color.Paint> errorBarColorSeriesNameToPaintMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.gui.color.Paint> errorBarColorSeriesNameToPaintMap() {
        return errorBarColorSeriesNameToPaintMap;
    }
    @Override public MultiXYSeriesSwappable errorBarColor(final io.deephaven.gui.color.Paint color, final Object... keys) {
        if(keys == null || keys.length == 0) {
            errorBarColorSeriesNameToPaintMap.setDefault(color);
        } else {
            errorBarColorSeriesNameToPaintMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                color);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Boolean> gradientVisibleSeriesNameTobooleanMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Boolean> gradientVisibleSeriesNameTobooleanMap() {
        return gradientVisibleSeriesNameTobooleanMap;
    }
    @Override public MultiXYSeriesSwappable gradientVisible(final boolean visible, final Object... keys) {
        if(keys == null || keys.length == 0) {
            gradientVisibleSeriesNameTobooleanMap.setDefault(visible);
        } else {
            gradientVisibleSeriesNameTobooleanMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                visible);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> lineColorSeriesNameToStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> lineColorSeriesNameToStringMap() {
        return lineColorSeriesNameToStringMap;
    }
    @Override public MultiXYSeriesSwappable lineColor(final java.lang.String color, final Object... keys) {
        if(keys == null || keys.length == 0) {
            lineColorSeriesNameToStringMap.setDefault(color);
        } else {
            lineColorSeriesNameToStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                color);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Integer> lineColorSeriesNameTointMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Integer> lineColorSeriesNameTointMap() {
        return lineColorSeriesNameTointMap;
    }
    @Override public MultiXYSeriesSwappable lineColor(final int color, final Object... keys) {
        if(keys == null || keys.length == 0) {
            lineColorSeriesNameTointMap.setDefault(color);
        } else {
            lineColorSeriesNameTointMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                color);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.gui.color.Paint> lineColorSeriesNameToPaintMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.gui.color.Paint> lineColorSeriesNameToPaintMap() {
        return lineColorSeriesNameToPaintMap;
    }
    @Override public MultiXYSeriesSwappable lineColor(final io.deephaven.gui.color.Paint color, final Object... keys) {
        if(keys == null || keys.length == 0) {
            lineColorSeriesNameToPaintMap.setDefault(color);
        } else {
            lineColorSeriesNameToPaintMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                color);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.db.plot.LineStyle> lineStyleSeriesNameToLineStyleMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.db.plot.LineStyle> lineStyleSeriesNameToLineStyleMap() {
        return lineStyleSeriesNameToLineStyleMap;
    }
    @Override public MultiXYSeriesSwappable lineStyle(final io.deephaven.db.plot.LineStyle style, final Object... keys) {
        if(keys == null || keys.length == 0) {
            lineStyleSeriesNameToLineStyleMap.setDefault(style);
        } else {
            lineStyleSeriesNameToLineStyleMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                style);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Boolean> linesVisibleSeriesNameToBooleanMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Boolean> linesVisibleSeriesNameToBooleanMap() {
        return linesVisibleSeriesNameToBooleanMap;
    }
    @Override public MultiXYSeriesSwappable linesVisible(final java.lang.Boolean visible, final Object... keys) {
        if(keys == null || keys.length == 0) {
            linesVisibleSeriesNameToBooleanMap.setDefault(visible);
        } else {
            linesVisibleSeriesNameToBooleanMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                visible);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, int[]> pointColorSeriesNameTointArrayMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, int[]> pointColorSeriesNameTointArrayMap() {
        return pointColorSeriesNameTointArrayMap;
    }
    @Override public MultiXYSeriesSwappable pointColor(final int[] colors, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointColorSeriesNameTointArrayMap.setDefault(colors);
        } else {
            pointColorSeriesNameTointArrayMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                colors);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.gui.color.Paint[]> pointColorSeriesNameToPaintArrayMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.gui.color.Paint[]> pointColorSeriesNameToPaintArrayMap() {
        return pointColorSeriesNameToPaintArrayMap;
    }
    @Override public MultiXYSeriesSwappable pointColor(final io.deephaven.gui.color.Paint[] colors, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointColorSeriesNameToPaintArrayMap.setDefault(colors);
        } else {
            pointColorSeriesNameToPaintArrayMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                colors);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Integer[]> pointColorSeriesNameToIntegerArrayMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Integer[]> pointColorSeriesNameToIntegerArrayMap() {
        return pointColorSeriesNameToIntegerArrayMap;
    }
    @Override public MultiXYSeriesSwappable pointColor(final java.lang.Integer[] colors, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointColorSeriesNameToIntegerArrayMap.setDefault(colors);
        } else {
            pointColorSeriesNameToIntegerArrayMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                colors);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String[]> pointColorSeriesNameToStringArrayMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String[]> pointColorSeriesNameToStringArrayMap() {
        return pointColorSeriesNameToStringArrayMap;
    }
    @Override public MultiXYSeriesSwappable pointColor(final java.lang.String[] colors, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointColorSeriesNameToStringArrayMap.setDefault(colors);
        } else {
            pointColorSeriesNameToStringArrayMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                colors);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> pointColorSeriesNameToStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> pointColorSeriesNameToStringMap() {
        return pointColorSeriesNameToStringMap;
    }
    @Override public MultiXYSeriesSwappable pointColor(final java.lang.String color, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointColorSeriesNameToStringMap.setDefault(color);
        } else {
            pointColorSeriesNameToStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                color);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Integer> pointColorSeriesNameTointMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Integer> pointColorSeriesNameTointMap() {
        return pointColorSeriesNameTointMap;
    }
    @Override public MultiXYSeriesSwappable pointColor(final int color, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointColorSeriesNameTointMap.setDefault(color);
        } else {
            pointColorSeriesNameTointMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                color);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.gui.color.Paint> pointColorSeriesNameToPaintMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.gui.color.Paint> pointColorSeriesNameToPaintMap() {
        return pointColorSeriesNameToPaintMap;
    }
    @Override public MultiXYSeriesSwappable pointColor(final io.deephaven.gui.color.Paint color, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointColorSeriesNameToPaintMap.setDefault(color);
        } else {
            pointColorSeriesNameToPaintMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                color);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.db.plot.datasets.data.IndexableData> pointColorSeriesNameToIndexableDataMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.db.plot.datasets.data.IndexableData> pointColorSeriesNameToIndexableDataMap() {
        return pointColorSeriesNameToIndexableDataMap;
    }
    @Override public <T extends io.deephaven.gui.color.Paint> MultiXYSeriesSwappable pointColor(final io.deephaven.db.plot.datasets.data.IndexableData<T> colors, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointColorSeriesNameToIndexableDataMap.setDefault(colors);
        } else {
            pointColorSeriesNameToIndexableDataMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                colors);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointColorSeriesNameToSelectableDataSetStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointColorSeriesNameToSelectableDataSetStringMap() {
        return pointColorSeriesNameToSelectableDataSetStringMap;
    }
    @Override public MultiXYSeriesSwappable pointColor(final io.deephaven.db.plot.filters.SelectableDataSet sds, final java.lang.String columnName, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointColorSeriesNameToSelectableDataSetStringMap.setDefault(new Object[]{sds, columnName});
        } else {
            pointColorSeriesNameToSelectableDataSetStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                new Object[]{ sds, columnName});
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointColorSeriesNameToTableStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointColorSeriesNameToTableStringMap() {
        return pointColorSeriesNameToTableStringMap;
    }
    @Override public MultiXYSeriesSwappable pointColor(final io.deephaven.db.tables.Table t, final java.lang.String columnName, final Object... keys) {
    final io.deephaven.db.plot.util.tables.TableHandle tHandle = new io.deephaven.db.plot.util.tables.TableHandle(t, columnName);
    addTableHandle(tHandle);
        if(keys == null || keys.length == 0) {
            pointColorSeriesNameToTableStringMap.setDefault(new Object[]{tHandle, columnName});
        } else {
            pointColorSeriesNameToTableStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                new Object[]{ tHandle, columnName});
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.db.plot.datasets.data.IndexableData> pointColorIntegerSeriesNameToIndexableDataMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.db.plot.datasets.data.IndexableData> pointColorIntegerSeriesNameToIndexableDataMap() {
        return pointColorIntegerSeriesNameToIndexableDataMap;
    }
    @Override public MultiXYSeriesSwappable pointColorInteger(final io.deephaven.db.plot.datasets.data.IndexableData<java.lang.Integer> colors, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointColorIntegerSeriesNameToIndexableDataMap.setDefault(colors);
        } else {
            pointColorIntegerSeriesNameToIndexableDataMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                colors);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointLabelSeriesNameToObjectArrayMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointLabelSeriesNameToObjectArrayMap() {
        return pointLabelSeriesNameToObjectArrayMap;
    }
    @Override public MultiXYSeriesSwappable pointLabel(final java.lang.Object[] labels, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointLabelSeriesNameToObjectArrayMap.setDefault(new Object[]{labels});
        } else {
            pointLabelSeriesNameToObjectArrayMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                new Object[]{labels});
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object> pointLabelSeriesNameToObjectMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object> pointLabelSeriesNameToObjectMap() {
        return pointLabelSeriesNameToObjectMap;
    }
    @Override public MultiXYSeriesSwappable pointLabel(final java.lang.Object label, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointLabelSeriesNameToObjectMap.setDefault(label);
        } else {
            pointLabelSeriesNameToObjectMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                label);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.db.plot.datasets.data.IndexableData> pointLabelSeriesNameToIndexableDataMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.db.plot.datasets.data.IndexableData> pointLabelSeriesNameToIndexableDataMap() {
        return pointLabelSeriesNameToIndexableDataMap;
    }
    @Override public MultiXYSeriesSwappable pointLabel(final io.deephaven.db.plot.datasets.data.IndexableData<?> labels, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointLabelSeriesNameToIndexableDataMap.setDefault(labels);
        } else {
            pointLabelSeriesNameToIndexableDataMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                labels);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointLabelSeriesNameToSelectableDataSetStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointLabelSeriesNameToSelectableDataSetStringMap() {
        return pointLabelSeriesNameToSelectableDataSetStringMap;
    }
    @Override public MultiXYSeriesSwappable pointLabel(final io.deephaven.db.plot.filters.SelectableDataSet sds, final java.lang.String columnName, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointLabelSeriesNameToSelectableDataSetStringMap.setDefault(new Object[]{sds, columnName});
        } else {
            pointLabelSeriesNameToSelectableDataSetStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                new Object[]{ sds, columnName});
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointLabelSeriesNameToTableStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointLabelSeriesNameToTableStringMap() {
        return pointLabelSeriesNameToTableStringMap;
    }
    @Override public MultiXYSeriesSwappable pointLabel(final io.deephaven.db.tables.Table t, final java.lang.String columnName, final Object... keys) {
    final io.deephaven.db.plot.util.tables.TableHandle tHandle = new io.deephaven.db.plot.util.tables.TableHandle(t, columnName);
    addTableHandle(tHandle);
        if(keys == null || keys.length == 0) {
            pointLabelSeriesNameToTableStringMap.setDefault(new Object[]{tHandle, columnName});
        } else {
            pointLabelSeriesNameToTableStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                new Object[]{ tHandle, columnName});
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> pointLabelFormatSeriesNameToStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> pointLabelFormatSeriesNameToStringMap() {
        return pointLabelFormatSeriesNameToStringMap;
    }
    @Override public MultiXYSeriesSwappable pointLabelFormat(final java.lang.String format, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointLabelFormatSeriesNameToStringMap.setDefault(format);
        } else {
            pointLabelFormatSeriesNameToStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                format);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.gui.shape.Shape[]> pointShapeSeriesNameToShapeArrayMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.gui.shape.Shape[]> pointShapeSeriesNameToShapeArrayMap() {
        return pointShapeSeriesNameToShapeArrayMap;
    }
    @Override public MultiXYSeriesSwappable pointShape(final io.deephaven.gui.shape.Shape[] shapes, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointShapeSeriesNameToShapeArrayMap.setDefault(shapes);
        } else {
            pointShapeSeriesNameToShapeArrayMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                shapes);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String[]> pointShapeSeriesNameToStringArrayMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String[]> pointShapeSeriesNameToStringArrayMap() {
        return pointShapeSeriesNameToStringArrayMap;
    }
    @Override public MultiXYSeriesSwappable pointShape(final java.lang.String[] shapes, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointShapeSeriesNameToStringArrayMap.setDefault(shapes);
        } else {
            pointShapeSeriesNameToStringArrayMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                shapes);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> pointShapeSeriesNameToStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> pointShapeSeriesNameToStringMap() {
        return pointShapeSeriesNameToStringMap;
    }
    @Override public MultiXYSeriesSwappable pointShape(final java.lang.String shape, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointShapeSeriesNameToStringMap.setDefault(shape);
        } else {
            pointShapeSeriesNameToStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                shape);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.gui.shape.Shape> pointShapeSeriesNameToShapeMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.gui.shape.Shape> pointShapeSeriesNameToShapeMap() {
        return pointShapeSeriesNameToShapeMap;
    }
    @Override public MultiXYSeriesSwappable pointShape(final io.deephaven.gui.shape.Shape shape, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointShapeSeriesNameToShapeMap.setDefault(shape);
        } else {
            pointShapeSeriesNameToShapeMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                shape);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.db.plot.datasets.data.IndexableData> pointShapeSeriesNameToIndexableDataMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.db.plot.datasets.data.IndexableData> pointShapeSeriesNameToIndexableDataMap() {
        return pointShapeSeriesNameToIndexableDataMap;
    }
    @Override public MultiXYSeriesSwappable pointShape(final io.deephaven.db.plot.datasets.data.IndexableData<java.lang.String> shapes, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointShapeSeriesNameToIndexableDataMap.setDefault(shapes);
        } else {
            pointShapeSeriesNameToIndexableDataMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                shapes);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointShapeSeriesNameToSelectableDataSetStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointShapeSeriesNameToSelectableDataSetStringMap() {
        return pointShapeSeriesNameToSelectableDataSetStringMap;
    }
    @Override public MultiXYSeriesSwappable pointShape(final io.deephaven.db.plot.filters.SelectableDataSet sds, final java.lang.String columnName, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointShapeSeriesNameToSelectableDataSetStringMap.setDefault(new Object[]{sds, columnName});
        } else {
            pointShapeSeriesNameToSelectableDataSetStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                new Object[]{ sds, columnName});
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointShapeSeriesNameToTableStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointShapeSeriesNameToTableStringMap() {
        return pointShapeSeriesNameToTableStringMap;
    }
    @Override public MultiXYSeriesSwappable pointShape(final io.deephaven.db.tables.Table t, final java.lang.String columnName, final Object... keys) {
    final io.deephaven.db.plot.util.tables.TableHandle tHandle = new io.deephaven.db.plot.util.tables.TableHandle(t, columnName);
    addTableHandle(tHandle);
        if(keys == null || keys.length == 0) {
            pointShapeSeriesNameToTableStringMap.setDefault(new Object[]{tHandle, columnName});
        } else {
            pointShapeSeriesNameToTableStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                new Object[]{ tHandle, columnName});
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object> pointSizeSeriesNameToTArrayMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object> pointSizeSeriesNameToTArrayMap() {
        return pointSizeSeriesNameToTArrayMap;
    }
    @Override public <T extends java.lang.Number> MultiXYSeriesSwappable pointSize(final T[] factors, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointSizeSeriesNameToTArrayMap.setDefault(factors);
        } else {
            pointSizeSeriesNameToTArrayMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                factors);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, double[]> pointSizeSeriesNameTodoubleArrayMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, double[]> pointSizeSeriesNameTodoubleArrayMap() {
        return pointSizeSeriesNameTodoubleArrayMap;
    }
    @Override public MultiXYSeriesSwappable pointSize(final double[] factors, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointSizeSeriesNameTodoubleArrayMap.setDefault(factors);
        } else {
            pointSizeSeriesNameTodoubleArrayMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                factors);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, int[]> pointSizeSeriesNameTointArrayMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, int[]> pointSizeSeriesNameTointArrayMap() {
        return pointSizeSeriesNameTointArrayMap;
    }
    @Override public MultiXYSeriesSwappable pointSize(final int[] factors, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointSizeSeriesNameTointArrayMap.setDefault(factors);
        } else {
            pointSizeSeriesNameTointArrayMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                factors);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, long[]> pointSizeSeriesNameTolongArrayMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, long[]> pointSizeSeriesNameTolongArrayMap() {
        return pointSizeSeriesNameTolongArrayMap;
    }
    @Override public MultiXYSeriesSwappable pointSize(final long[] factors, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointSizeSeriesNameTolongArrayMap.setDefault(factors);
        } else {
            pointSizeSeriesNameTolongArrayMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                factors);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Number> pointSizeSeriesNameToNumberMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Number> pointSizeSeriesNameToNumberMap() {
        return pointSizeSeriesNameToNumberMap;
    }
    @Override public MultiXYSeriesSwappable pointSize(final java.lang.Number factor, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointSizeSeriesNameToNumberMap.setDefault(factor);
        } else {
            pointSizeSeriesNameToNumberMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                factor);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.db.plot.datasets.data.IndexableData> pointSizeSeriesNameToIndexableDataMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.db.plot.datasets.data.IndexableData> pointSizeSeriesNameToIndexableDataMap() {
        return pointSizeSeriesNameToIndexableDataMap;
    }
    @Override public MultiXYSeriesSwappable pointSize(final io.deephaven.db.plot.datasets.data.IndexableData<java.lang.Double> factors, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointSizeSeriesNameToIndexableDataMap.setDefault(factors);
        } else {
            pointSizeSeriesNameToIndexableDataMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                factors);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointSizeSeriesNameToSelectableDataSetStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointSizeSeriesNameToSelectableDataSetStringMap() {
        return pointSizeSeriesNameToSelectableDataSetStringMap;
    }
    @Override public MultiXYSeriesSwappable pointSize(final io.deephaven.db.plot.filters.SelectableDataSet sds, final java.lang.String columnName, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointSizeSeriesNameToSelectableDataSetStringMap.setDefault(new Object[]{sds, columnName});
        } else {
            pointSizeSeriesNameToSelectableDataSetStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                new Object[]{ sds, columnName});
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointSizeSeriesNameToTableStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Object[]> pointSizeSeriesNameToTableStringMap() {
        return pointSizeSeriesNameToTableStringMap;
    }
    @Override public MultiXYSeriesSwappable pointSize(final io.deephaven.db.tables.Table t, final java.lang.String columnName, final Object... keys) {
    final io.deephaven.db.plot.util.tables.TableHandle tHandle = new io.deephaven.db.plot.util.tables.TableHandle(t, columnName);
    addTableHandle(tHandle);
        if(keys == null || keys.length == 0) {
            pointSizeSeriesNameToTableStringMap.setDefault(new Object[]{tHandle, columnName});
        } else {
            pointSizeSeriesNameToTableStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                new Object[]{ tHandle, columnName});
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Boolean> pointsVisibleSeriesNameToBooleanMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Boolean> pointsVisibleSeriesNameToBooleanMap() {
        return pointsVisibleSeriesNameToBooleanMap;
    }
    @Override public MultiXYSeriesSwappable pointsVisible(final java.lang.Boolean visible, final Object... keys) {
        if(keys == null || keys.length == 0) {
            pointsVisibleSeriesNameToBooleanMap.setDefault(visible);
        } else {
            pointsVisibleSeriesNameToBooleanMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                visible);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> seriesColorSeriesNameToStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> seriesColorSeriesNameToStringMap() {
        return seriesColorSeriesNameToStringMap;
    }
    @Override public MultiXYSeriesSwappable seriesColor(final java.lang.String color, final Object... keys) {
        if(keys == null || keys.length == 0) {
            seriesColorSeriesNameToStringMap.setDefault(color);
        } else {
            seriesColorSeriesNameToStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                color);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Integer> seriesColorSeriesNameTointMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.Integer> seriesColorSeriesNameTointMap() {
        return seriesColorSeriesNameTointMap;
    }
    @Override public MultiXYSeriesSwappable seriesColor(final int color, final Object... keys) {
        if(keys == null || keys.length == 0) {
            seriesColorSeriesNameTointMap.setDefault(color);
        } else {
            seriesColorSeriesNameTointMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                color);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.gui.color.Paint> seriesColorSeriesNameToPaintMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, io.deephaven.gui.color.Paint> seriesColorSeriesNameToPaintMap() {
        return seriesColorSeriesNameToPaintMap;
    }
    @Override public MultiXYSeriesSwappable seriesColor(final io.deephaven.gui.color.Paint color, final Object... keys) {
        if(keys == null || keys.length == 0) {
            seriesColorSeriesNameToPaintMap.setDefault(color);
        } else {
            seriesColorSeriesNameToPaintMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                color);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> toolTipPatternSeriesNameToStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> toolTipPatternSeriesNameToStringMap() {
        return toolTipPatternSeriesNameToStringMap;
    }
    @Override public MultiXYSeriesSwappable toolTipPattern(final java.lang.String format, final Object... keys) {
        if(keys == null || keys.length == 0) {
            toolTipPatternSeriesNameToStringMap.setDefault(format);
        } else {
            toolTipPatternSeriesNameToStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                format);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> xToolTipPatternSeriesNameToStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> xToolTipPatternSeriesNameToStringMap() {
        return xToolTipPatternSeriesNameToStringMap;
    }
    @Override public MultiXYSeriesSwappable xToolTipPattern(final java.lang.String format, final Object... keys) {
        if(keys == null || keys.length == 0) {
            xToolTipPatternSeriesNameToStringMap.setDefault(format);
        } else {
            xToolTipPatternSeriesNameToStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                format);
        }

        return this;
    }



    private io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> yToolTipPatternSeriesNameToStringMap = new io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<>();
    public io.deephaven.db.plot.util.PlotUtils.HashMapWithDefault<String, java.lang.String> yToolTipPatternSeriesNameToStringMap() {
        return yToolTipPatternSeriesNameToStringMap;
    }
    @Override public MultiXYSeriesSwappable yToolTipPattern(final java.lang.String format, final Object... keys) {
        if(keys == null || keys.length == 0) {
            yToolTipPatternSeriesNameToStringMap.setDefault(format);
        } else {
            yToolTipPatternSeriesNameToStringMap.put(namingFunction.apply(keys.length == 1 ? keys[0] : new io.deephaven.datastructures.util.SmartKey(keys)), 
                format);
        }

        return this;
    }



    @SuppressWarnings("unchecked") 
    private <T extends io.deephaven.gui.color.Paint, T0 extends java.lang.Number> void $$initializeSeries$$(XYDataSeriesInternal series) {
        String name = series.name().toString();
        java.util.function.Consumer<java.lang.Object[]> consumer0 = series::pointLabel;
        pointLabelSeriesNameToObjectArrayMap.runIfKeyExistsCast(consumer0, name);
        java.util.function.Consumer<java.lang.String> consumer1 = series::pointLabelFormat;
        pointLabelFormatSeriesNameToStringMap.runIfKeyExistsCast(consumer1, name);
        java.util.function.Consumer<io.deephaven.db.plot.datasets.data.IndexableData> consumer2 = series::pointShape;
        pointShapeSeriesNameToIndexableDataMap.runIfKeyExistsCast(consumer2, name);
        java.util.function.Consumer<java.lang.String[]> consumer3 = series::pointShape;
        pointShapeSeriesNameToStringArrayMap.runIfKeyExistsCast(consumer3, name);
        java.lang.Object[]         objectArray = pointColorSeriesNameToTableStringMap.get(name);
        if(objectArray != null) {series.pointColor(((io.deephaven.db.plot.util.tables.TableHandle) objectArray[0]).getTable(), (java.lang.String) objectArray[1]);}

        objectArray = pointSizeSeriesNameToSelectableDataSetStringMap.get(name);
        if(objectArray != null) {series.pointSize((io.deephaven.db.plot.filters.SelectableDataSet) objectArray[0], (java.lang.String) objectArray[1]);}

        java.util.function.Consumer<java.lang.Boolean> consumer4 = series::pointsVisible;
        pointsVisibleSeriesNameToBooleanMap.runIfKeyExistsCast(consumer4, name);
        java.util.function.Consumer<io.deephaven.db.plot.datasets.data.IndexableData> consumer5 = series::pointColor;
        pointColorSeriesNameToIndexableDataMap.runIfKeyExistsCast(consumer5, name);
        objectArray = pointShapeSeriesNameToSelectableDataSetStringMap.get(name);
        if(objectArray != null) {series.pointShape((io.deephaven.db.plot.filters.SelectableDataSet) objectArray[0], (java.lang.String) objectArray[1]);}

        objectArray = pointLabelSeriesNameToTableStringMap.get(name);
        if(objectArray != null) {series.pointLabel(((io.deephaven.db.plot.util.tables.TableHandle) objectArray[0]).getTable(), (java.lang.String) objectArray[1]);}

        java.util.function.Consumer<java.lang.String> consumer6 = series::pointColor;
        pointColorSeriesNameToStringMap.runIfKeyExistsCast(consumer6, name);
        java.util.function.Consumer<io.deephaven.gui.shape.Shape> consumer7 = series::pointShape;
        pointShapeSeriesNameToShapeMap.runIfKeyExistsCast(consumer7, name);
        java.util.function.Consumer<java.lang.Integer> consumer8 = series::lineColor;
        lineColorSeriesNameTointMap.runIfKeyExistsCast(consumer8, name);
        java.util.function.Consumer<java.lang.Boolean> consumer9 = series::linesVisible;
        linesVisibleSeriesNameToBooleanMap.runIfKeyExistsCast(consumer9, name);
        objectArray = pointLabelSeriesNameToSelectableDataSetStringMap.get(name);
        if(objectArray != null) {series.pointLabel((io.deephaven.db.plot.filters.SelectableDataSet) objectArray[0], (java.lang.String) objectArray[1]);}

        java.util.function.Consumer<java.lang.String> consumer10 = series::lineColor;
        lineColorSeriesNameToStringMap.runIfKeyExistsCast(consumer10, name);
        java.util.function.Consumer<io.deephaven.gui.color.Paint> consumer11 = series::seriesColor;
        seriesColorSeriesNameToPaintMap.runIfKeyExistsCast(consumer11, name);
        java.util.function.Consumer<long[]> consumer12 = series::pointSize;
        pointSizeSeriesNameTolongArrayMap.runIfKeyExistsCast(consumer12, name);
        java.util.function.Consumer<io.deephaven.gui.shape.Shape[]> consumer13 = series::pointShape;
        pointShapeSeriesNameToShapeArrayMap.runIfKeyExistsCast(consumer13, name);
        java.util.function.Consumer<java.lang.String> consumer14 = series::yToolTipPattern;
        yToolTipPatternSeriesNameToStringMap.runIfKeyExistsCast(consumer14, name);
        java.util.function.Consumer<java.lang.Integer> consumer15 = series::seriesColor;
        seriesColorSeriesNameTointMap.runIfKeyExistsCast(consumer15, name);
        java.util.function.Consumer<int[]> consumer16 = series::pointColor;
        pointColorSeriesNameTointArrayMap.runIfKeyExistsCast(consumer16, name);
        java.util.function.Consumer<java.lang.Object> consumer17 = series::pointLabel;
        pointLabelSeriesNameToObjectMap.runIfKeyExistsCast(consumer17, name);
        java.util.function.Consumer<io.deephaven.gui.color.Paint> consumer18 = series::errorBarColor;
        errorBarColorSeriesNameToPaintMap.runIfKeyExistsCast(consumer18, name);
        objectArray = pointColorSeriesNameToSelectableDataSetStringMap.get(name);
        if(objectArray != null) {series.pointColor((io.deephaven.db.plot.filters.SelectableDataSet) objectArray[0], (java.lang.String) objectArray[1]);}

        java.util.function.Consumer<io.deephaven.gui.color.Paint> consumer19 = series::lineColor;
        lineColorSeriesNameToPaintMap.runIfKeyExistsCast(consumer19, name);
        java.util.function.Consumer<java.lang.String> consumer20 = series::seriesColor;
        seriesColorSeriesNameToStringMap.runIfKeyExistsCast(consumer20, name);
        java.util.function.Consumer<java.lang.Integer[]> consumer21 = series::pointColor;
        pointColorSeriesNameToIntegerArrayMap.runIfKeyExistsCast(consumer21, name);
        java.util.function.Consumer<io.deephaven.db.plot.LineStyle> consumer22 = series::lineStyle;
        lineStyleSeriesNameToLineStyleMap.runIfKeyExistsCast(consumer22, name);
        java.util.function.Consumer<double[]> consumer23 = series::pointSize;
        pointSizeSeriesNameTodoubleArrayMap.runIfKeyExistsCast(consumer23, name);
        java.util.function.Consumer<java.lang.Boolean> consumer24 = series::gradientVisible;
        gradientVisibleSeriesNameTobooleanMap.runIfKeyExistsCast(consumer24, name);
        java.util.function.Consumer<io.deephaven.gui.color.Paint> consumer25 = series::pointColor;
        pointColorSeriesNameToPaintMap.runIfKeyExistsCast(consumer25, name);
        java.util.function.Consumer<io.deephaven.db.plot.datasets.data.IndexableData> consumer26 = series::pointColorInteger;
        pointColorIntegerSeriesNameToIndexableDataMap.runIfKeyExistsCast(consumer26, name);
        java.util.function.Consumer<int[]> consumer27 = series::pointSize;
        pointSizeSeriesNameTointArrayMap.runIfKeyExistsCast(consumer27, name);
        java.util.function.Consumer<java.lang.String[]> consumer28 = series::pointColor;
        pointColorSeriesNameToStringArrayMap.runIfKeyExistsCast(consumer28, name);
        java.util.function.Consumer<T0[]> consumer29 = series::pointSize;
        pointSizeSeriesNameToTArrayMap.runIfKeyExistsCast(consumer29, name);
        java.util.function.Consumer<io.deephaven.db.plot.datasets.data.IndexableData> consumer30 = series::pointLabel;
        pointLabelSeriesNameToIndexableDataMap.runIfKeyExistsCast(consumer30, name);
        objectArray = pointShapeSeriesNameToTableStringMap.get(name);
        if(objectArray != null) {series.pointShape(((io.deephaven.db.plot.util.tables.TableHandle) objectArray[0]).getTable(), (java.lang.String) objectArray[1]);}

        java.util.function.Consumer<java.lang.Number> consumer31 = series::pointSize;
        pointSizeSeriesNameToNumberMap.runIfKeyExistsCast(consumer31, name);
        java.util.function.Consumer<java.lang.String> consumer32 = series::errorBarColor;
        errorBarColorSeriesNameToStringMap.runIfKeyExistsCast(consumer32, name);
        java.util.function.Consumer<java.lang.Integer> consumer33 = series::pointColor;
        pointColorSeriesNameTointMap.runIfKeyExistsCast(consumer33, name);
        java.util.function.Consumer<io.deephaven.db.plot.datasets.data.IndexableData> consumer34 = series::pointSize;
        pointSizeSeriesNameToIndexableDataMap.runIfKeyExistsCast(consumer34, name);
        java.util.function.Consumer<java.lang.String> consumer35 = series::xToolTipPattern;
        xToolTipPatternSeriesNameToStringMap.runIfKeyExistsCast(consumer35, name);
        java.util.function.Consumer<java.lang.Integer> consumer36 = series::errorBarColor;
        errorBarColorSeriesNameTointMap.runIfKeyExistsCast(consumer36, name);
        java.util.function.Consumer<io.deephaven.gui.color.Paint[]> consumer37 = series::pointColor;
        pointColorSeriesNameToPaintArrayMap.runIfKeyExistsCast(consumer37, name);
        objectArray = pointSizeSeriesNameToTableStringMap.get(name);
        if(objectArray != null) {series.pointSize(((io.deephaven.db.plot.util.tables.TableHandle) objectArray[0]).getTable(), (java.lang.String) objectArray[1]);}

        java.util.function.Consumer<java.lang.String> consumer38 = series::pointShape;
        pointShapeSeriesNameToStringMap.runIfKeyExistsCast(consumer38, name);
        java.util.function.Consumer<java.lang.String> consumer39 = series::toolTipPattern;
        toolTipPatternSeriesNameToStringMap.runIfKeyExistsCast(consumer39, name);

    }
    @Override
    public MultiXYSeriesSwappable copy(AxesImpl axes) {
        final MultiXYSeriesSwappable __s__ = new MultiXYSeriesSwappable(this, axes);
                __s__.pointLabelSeriesNameToObjectArrayMap = pointLabelSeriesNameToObjectArrayMap.copy();
        __s__.pointLabelFormatSeriesNameToStringMap = pointLabelFormatSeriesNameToStringMap.copy();
        __s__.pointShapeSeriesNameToIndexableDataMap = pointShapeSeriesNameToIndexableDataMap.copy();
        __s__.pointShapeSeriesNameToStringArrayMap = pointShapeSeriesNameToStringArrayMap.copy();
        __s__.pointColorSeriesNameToTableStringMap = pointColorSeriesNameToTableStringMap.copy();
        __s__.pointSizeSeriesNameToSelectableDataSetStringMap = pointSizeSeriesNameToSelectableDataSetStringMap.copy();
        __s__.pointsVisibleSeriesNameToBooleanMap = pointsVisibleSeriesNameToBooleanMap.copy();
        __s__.pointColorSeriesNameToIndexableDataMap = pointColorSeriesNameToIndexableDataMap.copy();
        __s__.pointShapeSeriesNameToSelectableDataSetStringMap = pointShapeSeriesNameToSelectableDataSetStringMap.copy();
        __s__.pointLabelSeriesNameToTableStringMap = pointLabelSeriesNameToTableStringMap.copy();
        __s__.pointColorSeriesNameToStringMap = pointColorSeriesNameToStringMap.copy();
        __s__.pointShapeSeriesNameToShapeMap = pointShapeSeriesNameToShapeMap.copy();
        __s__.lineColorSeriesNameTointMap = lineColorSeriesNameTointMap.copy();
        __s__.linesVisibleSeriesNameToBooleanMap = linesVisibleSeriesNameToBooleanMap.copy();
        __s__.pointLabelSeriesNameToSelectableDataSetStringMap = pointLabelSeriesNameToSelectableDataSetStringMap.copy();
        __s__.lineColorSeriesNameToStringMap = lineColorSeriesNameToStringMap.copy();
        __s__.seriesColorSeriesNameToPaintMap = seriesColorSeriesNameToPaintMap.copy();
        __s__.pointSizeSeriesNameTolongArrayMap = pointSizeSeriesNameTolongArrayMap.copy();
        __s__.pointShapeSeriesNameToShapeArrayMap = pointShapeSeriesNameToShapeArrayMap.copy();
        __s__.yToolTipPatternSeriesNameToStringMap = yToolTipPatternSeriesNameToStringMap.copy();
        __s__.seriesColorSeriesNameTointMap = seriesColorSeriesNameTointMap.copy();
        __s__.pointColorSeriesNameTointArrayMap = pointColorSeriesNameTointArrayMap.copy();
        __s__.pointLabelSeriesNameToObjectMap = pointLabelSeriesNameToObjectMap.copy();
        __s__.errorBarColorSeriesNameToPaintMap = errorBarColorSeriesNameToPaintMap.copy();
        __s__.pointColorSeriesNameToSelectableDataSetStringMap = pointColorSeriesNameToSelectableDataSetStringMap.copy();
        __s__.lineColorSeriesNameToPaintMap = lineColorSeriesNameToPaintMap.copy();
        __s__.seriesColorSeriesNameToStringMap = seriesColorSeriesNameToStringMap.copy();
        __s__.pointColorSeriesNameToIntegerArrayMap = pointColorSeriesNameToIntegerArrayMap.copy();
        __s__.lineStyleSeriesNameToLineStyleMap = lineStyleSeriesNameToLineStyleMap.copy();
        __s__.pointSizeSeriesNameTodoubleArrayMap = pointSizeSeriesNameTodoubleArrayMap.copy();
        __s__.gradientVisibleSeriesNameTobooleanMap = gradientVisibleSeriesNameTobooleanMap.copy();
        __s__.pointColorSeriesNameToPaintMap = pointColorSeriesNameToPaintMap.copy();
        __s__.pointColorIntegerSeriesNameToIndexableDataMap = pointColorIntegerSeriesNameToIndexableDataMap.copy();
        __s__.pointSizeSeriesNameTointArrayMap = pointSizeSeriesNameTointArrayMap.copy();
        __s__.pointColorSeriesNameToStringArrayMap = pointColorSeriesNameToStringArrayMap.copy();
        __s__.pointSizeSeriesNameToTArrayMap = pointSizeSeriesNameToTArrayMap.copy();
        __s__.pointLabelSeriesNameToIndexableDataMap = pointLabelSeriesNameToIndexableDataMap.copy();
        __s__.pointShapeSeriesNameToTableStringMap = pointShapeSeriesNameToTableStringMap.copy();
        __s__.pointSizeSeriesNameToNumberMap = pointSizeSeriesNameToNumberMap.copy();
        __s__.errorBarColorSeriesNameToStringMap = errorBarColorSeriesNameToStringMap.copy();
        __s__.pointColorSeriesNameTointMap = pointColorSeriesNameTointMap.copy();
        __s__.pointSizeSeriesNameToIndexableDataMap = pointSizeSeriesNameToIndexableDataMap.copy();
        __s__.xToolTipPatternSeriesNameToStringMap = xToolTipPatternSeriesNameToStringMap.copy();
        __s__.errorBarColorSeriesNameTointMap = errorBarColorSeriesNameTointMap.copy();
        __s__.pointColorSeriesNameToPaintArrayMap = pointColorSeriesNameToPaintArrayMap.copy();
        __s__.pointSizeSeriesNameToTableStringMap = pointSizeSeriesNameToTableStringMap.copy();
        __s__.pointShapeSeriesNameToStringMap = pointShapeSeriesNameToStringMap.copy();
        __s__.toolTipPatternSeriesNameToStringMap = toolTipPatternSeriesNameToStringMap.copy();
        return __s__;
    }
}