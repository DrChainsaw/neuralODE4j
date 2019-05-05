package util.plot;

import java.awt.*;
import java.io.IOException;
import java.util.List;

/**
 * Does not plot anything
 * @param <X>
 * @param <Y>
 *
 * @author Christian Skarby
 */
public class NoPlot<X extends Number, Y extends Number> implements Plot<X, Y> {

    public static class Factory implements Plot.Factory {

        @Override
        public <X extends Number, Y extends Number> Plot<X, Y> newPlot(String title) {
            return new NoPlot<>();
        }
    }

    private static class Series implements Plot.Series {

        @Override
        public Plot.Series line() {
            return this;
        }

        @Override
        public Plot.Series scatter() {
            return this;
        }

        @Override
        public Plot.Series set(Color color) {
            return this;
        }
    }

    @Override
    public Series createSeries(String label) {
        return new Series();
    }

    @Override
    public Series plotData(String label, X x, Y y) {
        return new Series();
    }

    @Override
    public Series plotData(String label, List<X> x, List<Y> y) {
        return new Series();
    }

    @Override
    public Series clearData(String label) {
        return new Series();
    }

    @Override
    public void clearData() {
        // Ignore
    }

    @Override
    public void storePlotData() throws IOException {
        // Ignore
    }

    @Override
    public void storePlotData(String label) throws IOException {
        // Ignore
    }

    @Override
    public void savePicture(String suffix) throws IOException {
        // Ignore
    }
}
