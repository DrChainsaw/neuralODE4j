package examples.spiral.listener;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import util.plot.Plot;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Plots the decoded output
 *
 * @author Christian Skarby
 */
public class PlotDecodedOutput extends BaseTrainingListener {

    private final Plot<Double, Double> plot;
    private final String outputName;
    private final int batchNrToPlot;

    public PlotDecodedOutput(Plot<Double, Double> plot, String outputName, int batchNrToPlot) {
        this.plot = plot;
        this.outputName = outputName;
        this.batchNrToPlot = batchNrToPlot;
        plot.createSeries(outputName);
    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        final INDArray toPlot = activations.get(outputName).tensorAlongDimension(batchNrToPlot, 1,2);
        plot.clearData(outputName);
        final List<Double> x = toDoubleList(toPlot, 0);
        final List<Double> y = toDoubleList(toPlot, 1);
        plot.plotData(outputName, x, y);
    }

    private static List<Double> toDoubleList(INDArray toPlot, int row) {
        final List<Double> out = new ArrayList<>();
        for(double d: toPlot.getRow(row).toDoubleVector()) {
            out.add(d);
        }
        return out;
    }
}
