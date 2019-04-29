
package util.plot;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.deser.std.StdDeserializer;
import com.fasterxml.jackson.databind.module.SimpleModule;
import lombok.Data;
import org.jetbrains.annotations.NotNull;
import org.knowm.xchart.*;
import org.knowm.xchart.style.Styler;
import org.knowm.xchart.style.Styler.ChartTheme;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Real time updatable plot with support for an arbitrary number of series. Can also serialize the plotted data and
 * recreate a plot from such data. Typically used for plot training/eval metrics for each iteration. Note: The amount
 * of data points per timeseries is limited to 5000 as a significant slowdown was observed for higher numbers. When 5000
 * points is reached, all even points will be removed. New points after this will be added as normal until the total hits
 * 5000 again.
 *
 * @author Christian Skärby
 */
public class RealTimePlot<X extends Number, Y extends Number> implements Plot<X, Y> {

    private final String title;
    private final XYChart xyChart;
    private final SwingWrapper<XYChart> swingWrapper;
    private final String plotDir;

    private final Map<String, DataXY<X, Y>> plotSeries = new HashMap<>();

    /**
     * Factory for this class
     */
    public static class Factory implements Plot.Factory {

        private final String plotDir;

        public Factory(String plotDir) {
            this.plotDir = plotDir;
        }

        @Override
        public <X extends Number, Y extends Number> Plot<X, Y> newPlot(String title) {
            return new RealTimePlot<>(title, plotDir);
        }
    }

    @JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, property = "@class")
    @Data
    private static class DataXY<X extends Number, Y extends Number> implements Serializable {

        private static final long serialVersionUID = 7526471155622776891L;
        private final String series;

        private final LinkedList<X> xData;
        private final LinkedList<Y> yData;

        DataXY(String series) {
            this(series, new LinkedList<>(), new LinkedList<>());
        }

        DataXY(
                @JsonProperty("series") String series,
                @JsonProperty("xData") List<X> xData,
                @JsonProperty("yData") List<Y> yData) {
            this.series = series;
            this.xData = new LinkedList<>(xData);
            this.yData = new LinkedList<>(yData);
        }

        private void addPoint(X x, Y y, XYChart xyChart, SwingWrapper<XYChart> swingWrapper) {
            xData.addLast(x);
            yData.addLast(y);
            if (xData.size() > 5000) {
                for (int i = 0; i < xData.size(); i += 2) {
                    xData.remove(i);
                    yData.remove(i);
                }
            }
            plotData(xyChart, swingWrapper);
        }

        private void addData(List<X> x, List<Y> y, XYChart xyChart, SwingWrapper<XYChart> swingWrapper) {
            xData.addAll(x);
            yData.addAll(y);
            plotData(xyChart, swingWrapper);
        }

        private void plotData(XYChart xyChart, SwingWrapper<XYChart> swingWrapper) {
            javax.swing.SwingUtilities.invokeLater(() -> {
                if (!xyChart.getSeriesMap().containsKey(series)) {
                    xyChart.addSeries(series, xData, yData, null);
                } else {
                    xyChart.updateXYSeries(series, xData, yData, null);
                }
                swingWrapper.repaintChart();
            });
        }

        private void createSeries(final XYChart xyChart, SwingWrapper<XYChart> swingWrapper) {
            javax.swing.SwingUtilities.invokeLater(() -> {
                if (xData.size() == 0) {
                    xyChart.addSeries(series, Arrays.asList(0), Arrays.asList(1));
                } else {
                    xyChart.addSeries(series, xData, yData);
                }
                swingWrapper.repaintChart();
            });
        }


        private void clear() {
            xData.clear();
            yData.clear();
        }
    }

    private class DataXYDeserializer<X extends Number, Y extends Number> extends StdDeserializer<DataXY<X, Y>> {

        public DataXYDeserializer() {
            this(null);
        }

        public DataXYDeserializer(Class<?> vc) {
            super(vc);
        }

        @Override
        public DataXY<X, Y> deserialize(JsonParser jp, DeserializationContext ctxt)
                throws IOException {
            JsonNode node = jp.getCodec().readTree(jp);

            final List<X> xData = new ArrayList<>();
            for (JsonNode xNode : node.get("xdata")) {
                xData.add((X) xNode.numberValue());
            }

            final List<Y> yData = new ArrayList<>();
            for (JsonNode xNode : node.get("ydata")) {
                yData.add((Y) xNode.numberValue());
            }
            return new DataXY<>(node.get("series").toString(), xData, yData);
        }

    }

    private static class Series implements Plot.Series {

        private final XYChart chart;
        private final String label;

        public Series(XYChart chart, String label) {
            this.chart = chart;
            this.label = label;
        }

        @Override
        public Plot.Series line() {
            javax.swing.SwingUtilities.invokeLater(() -> series().setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line));
            return this;
        }

        @Override
        public Plot.Series scatter() {
            javax.swing.SwingUtilities.invokeLater(() -> series().setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter));
            return this;
        }

        @Override
        public Plot.Series set(Color color) {
            javax.swing.SwingUtilities.invokeLater(() -> {
                series().setLineColor(color);
                series().setMarkerColor(color);
                series().setFillColor(color);
            });
            return this;
        }

        private XYSeries series() {
            return chart.getSeriesMap().get(label);
        }
    }

    /**
     * Constructor
     *
     * @param title   Title of the plot
     * @param plotDir Directory to store plots in.
     */
    public RealTimePlot(String title, String plotDir) {
        // Create Chart
        this.title = title;
        xyChart = new XYChartBuilder().width(800).height(500).theme(ChartTheme.Matlab).title(title).build();
        xyChart.getStyler().setLegendPosition(Styler.LegendPosition.OutsideE);
        xyChart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);

        this.swingWrapper = new SwingWrapper<>(xyChart);
        swingWrapper.displayChart();
        this.plotDir = plotDir;
    }

    @Override
    public Series plotData(String label, X x, Y y) {
        final DataXY<X, Y> data = getOrCreateSeries(label);
        data.addPoint(x, y, xyChart, swingWrapper);
        return new Series(xyChart, label);
    }

    @Override
    public Series plotData(String label, List<X> x, List<Y> y) {
        final DataXY<X, Y> data = getOrCreateSeries(label);
        data.addData(x, y, xyChart, swingWrapper);
        return new Series(xyChart, label);
    }

    @Override
    public Series clearData(String label) {
        final DataXY<X, Y> data = getOrCreateSeries(label);
        data.clear();
        return new Series(xyChart, label);
    }

    @Override
    public Series createSeries(String label) {
        getOrCreateSeries(label);
        return new Series(xyChart, label);
    }

    @NotNull
    private DataXY<X, Y> getOrCreateSeries(String label) {
        DataXY<X, Y> data = plotSeries.get(label);
        if (data == null) {
            data = restoreOrCreatePlotData(label);
            plotSeries.put(label, data);
            data.createSeries(xyChart, swingWrapper);
        }
        return data;
    }

    @Override
    public void storePlotData() throws IOException {
        for (String label : plotSeries.keySet()) {
            storePlotData(label);
        }
    }

    @Override
    public void storePlotData(String label) throws IOException {
        DataXY<X, Y> data = plotSeries.get(label);
        if (data != null) {
            new ObjectMapper().writeValue(new File(createFileName(label)), data);
        }
    }

    @Override
    public void savePicture(String suffix) {
        SwingUtilities.invokeLater(() -> {
            try {
                BitmapEncoder.saveBitmap(xyChart, plotDir + File.separator + title + suffix, BitmapEncoder.BitmapFormat.PNG);
            } catch (IOException e) {
                throw new UncheckedIOException("Save picture in " + this.getClass() + " failed!", e);
            }
        });
    }

    private DataXY<X, Y> restoreOrCreatePlotData(String label) {
        File dataFile = new File(createFileName(label));
        if (dataFile.exists()) {
            try {
                final SimpleModule mod = new SimpleModule();
                mod.addDeserializer(DataXY.class, new DataXYDeserializer<X, Y>());
                final ObjectMapper mapper = new ObjectMapper().registerModule(mod);
                return mapper.readValue(dataFile, new TypeReference<DataXY<X, Y>>() {
                });
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return new DataXY<>(label);
    }

    private String createFileName(String label) {
        return plotDir + File.separator + title + "_" + label + ".plt";
    }

    public static void main(String[] args) {

        final RealTimePlot<Integer, Double> plotter = new RealTimePlot<>("Test plot", "");
        IntStream.range(1000, 2000).forEach(x -> Stream.of("s1", "s2", "s3").forEach(str -> {
            plotter.createSeries(str);
            plotter.plotData(str, x, 1d / ((double) x + 10));
        }));
//          try {
//            plotter.storePlotData("s1");
//          } catch (IOException e) {
//             e.printStackTrace();
//          }

    }
}