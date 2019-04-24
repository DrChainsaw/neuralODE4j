package util.plot;

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

    @Override
    public void createSeries(String label) {
        // Ignore
    }

    @Override
    public void plotData(String label, X x, Y y) {
        // Ignore
    }

    @Override
    public void plotData(String label, List<X> x, List<Y> y) {
        // Ignore
    }

    @Override
    public void clearData(String label) {
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
