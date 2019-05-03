package examples.anode;

import org.jzy3d.chart.Chart;
import org.jzy3d.chart.ChartLauncher;
import org.jzy3d.chart.Settings;
import org.jzy3d.chart.factories.AWTChartComponentFactory;
import org.jzy3d.chart.factories.IChartComponentFactory;
import org.jzy3d.colors.Color;
import org.jzy3d.maths.Coord3d;
import org.jzy3d.maths.Rectangle;
import org.jzy3d.plot3d.primitives.Scatter;
import org.jzy3d.plot3d.rendering.canvas.Quality;
import org.jzy3d.plot3d.rendering.view.modes.ViewBoundMode;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 3D scatter plot
 *
 * @author Christian Skarby
 */
class ScatterPlot3D implements Plot3D {

    private final Chart chart;
    private final String title;
    private final String baseDir;
    private final Map<String, DataSeries> seriesMap = new HashMap<>();

    private static class DataSeries implements Series3D {

        private final List<Coord3d> points = new ArrayList<>();
        private final Scatter scatter;
        private final Chart chart;

        private DataSeries(Chart chart) {
            this.chart = chart;
            this.scatter = new Scatter();
            scatter.setWidth(2);
            this.chart.getScene().getGraph().add(scatter);
        }

        public DataSeries plot(List<Double> x, List<Double> y, List<Double> z) {
            javax.swing.SwingUtilities.invokeLater(() -> {
                for (int i = 0; i < x.size(); i++) {
                    points.add(new Coord3d(x.get(i), y.get(i), z.get(i)));
                }
                update();
            });
            return this;
        }

        public DataSeries color(Color color) {
            javax.swing.SwingUtilities.invokeLater(() -> scatter.setColor(color));
            return this;
        }

        @Override
        public Series3D size(float size) {
            javax.swing.SwingUtilities.invokeLater(() -> scatter.setWidth(size));
            return this;
        }

        public DataSeries clear() {
            javax.swing.SwingUtilities.invokeLater(() -> {
                points.clear();
                update();
            });
            return this;
        }

        private void update() {
            scatter.setData(points.toArray(new Coord3d[0]));
            chart.getScene().getGraph().add(scatter, true);
        }
    }

    ScatterPlot3D(String title, String plotDir) {
        this.baseDir = plotDir;
        Settings.getInstance().setHardwareAccelerated(true);
        chart = AWTChartComponentFactory.chart(Quality.Advanced, IChartComponentFactory.Toolkit.newt);
        series("");

        ChartLauncher.openChart(chart, new Rectangle(0, 0, 600, 600), title);
        chart.getView().setBoundMode(ViewBoundMode.AUTO_FIT);

        this.title = title;
    }

    @Override
    public Series3D series(String label) {
        return seriesMap.computeIfAbsent(label, lab -> new DataSeries(chart));
    }

    @Override
    public void savePicture(String suffix) {
        javax.swing.SwingUtilities.invokeLater(() -> {
            try {
                chart.screenshot(Paths.get(baseDir, title + suffix + ".png").toFile());
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }

    @Override
    public void fit() {
        javax.swing.SwingUtilities.invokeLater(() -> chart.getView().setBoundMode(ViewBoundMode.AUTO_FIT));
    }
}
