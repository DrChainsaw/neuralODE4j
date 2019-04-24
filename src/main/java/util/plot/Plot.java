package util.plot;

import java.io.IOException;
import java.util.List;

/**
 * Interface for XY plotting. Supports serialization of plot data.
 * @param <X>
 * @param <Y>
 */
public interface Plot<X extends Number, Y extends Number> {

    /**
     * Factory interface
     */
    interface Factory {

        /**
         * Create a new plot with the given title
         * @param title title of the plot
         * @return a new Plot instance
         */
        <X extends Number, Y extends  Number> Plot<X, Y> newPlot(String title);
    }

    /**
     * Creates a time series for the given label. If data with the given label exists in serialized format in the
     * plotDir the time series of that data will be recreated.
     * @param label series label.
     */
    void createSeries(String label);

    /**
     * Plot some data belonging to a certain label. Will be appended to an existing series of such exists, either in
     * an existing window or in serialized format in the plotDir. If no series with the given label exists it will
     * be created in the window of this plot instance.
     * @param label series label
     * @param x point on x axis
     * @param y point on y axis
     */
    void plotData(String label, X x, Y y);

    /**
     * Plot some data belonging to a certain label. Will be appended to an existing series of such exists, either in
     * an existing window or in serialized format in the plotDir. If no series with the given label exists it will
     * be created in the window of this plot instance.
     * @param label series label
     * @param x points on x axis
     * @param y points on y axis
     */
    void plotData(String label, List<X> x, List<Y> y);

    /**
     * Clears the data for the given label
     * @param label series label
     */
    void clearData(String label);

    /**
     * Serialize the data for all labels.
     * @throws IOException
     */
    void storePlotData() throws IOException;

    /**
     * Serialize the data for the given label.
     * @param label series label
     * @throws IOException
     */
    void storePlotData(String label) throws IOException;

    /**
     * Save plot as a picture
     */
    void savePicture(String suffix) throws IOException;

    /**
     * Convenience method for debugging purposes. Plots the given data vs list indexes
     * @param data
     * @param <Y>
     */
    static <Y extends Number> void plot(List<Y> data, String plotName) {
        final RealTimePlot<Integer, Y> plotter = new RealTimePlot<>(plotName, "");
        plotter.createSeries("data");
        for(int x = 0; x < data.size(); x ++) {
            plotter.plotData("data", x, data.get(x));
        }
    }
}
