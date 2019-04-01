package examples.spiral.listener;

import org.nd4j.linalg.api.ndarray.INDArray;
import util.plot.Plot;

import java.util.ArrayList;
import java.util.List;

/**
 * Simple plot util for plotting spirals from a 2D {@link org.nd4j.linalg.api.ndarray.INDArray} where each column is is x
 * and y coordinates for a time index.
 *
 * @author Christian Skarby
 */
public class SpiralPlot {

    private final Plot<Double, Double> plot;

    public SpiralPlot(Plot<Double, Double> plot) {
        this.plot = plot;
    }

    /**
     * Create a label in the plot. Call this method before plotting anything in any label in the same plot to avoid
     * null pointer exceptions in the plot thread.
     * @param label Series to newPlot
     */
    public void createSeries(String label) {
        plot.createSeries(label);
    }

    /**
     * Plot data assuming each row of the given {@link INDArray} is an x and y coordinate pair.
     * @param label Label for curve to plot
     * @param toPlot Data to plot
     */
    public void plot(String label, INDArray toPlot) {
        plot.clearData(label);
        final List<Double> x = toDoubleList(toPlot, 0);
        final List<Double> y = toDoubleList(toPlot, 1);
        plot.plotData(label, x, y);
    }

    /**
     * Plot data assuming each a 3D {@link INDArray} where each element along the first dimension is a set of
     * x and y coordinate pairs.
     * @param label Label for curve to plot
     * @param toPlot Data to plot
     * @param batchNr which set of x,y pairs to plot
     */
    public void plot(String label, INDArray toPlot, int batchNr) {
        plot(label, toPlot.tensorAlongDimension(batchNr, 1,2));
    }

    private static List<Double> toDoubleList(INDArray toPlot, int row) {
        final List<Double> out = new ArrayList<>();
        for(double d: toPlot.getRow(row).toDoubleVector()) {
            out.add(d);
        }
        return out;
    }
}
