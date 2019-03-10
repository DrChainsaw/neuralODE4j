package examples.spiral.listener;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import util.plot.Plot;

import java.util.Map;

/**
 * Plots the decoded output
 *
 * @author Christian Skarby
 */
public class PlotDecodedOutput extends BaseTrainingListener {

    private final SpiralPlot plot;
    private final String outputName;
    private final String plotLabel;
    private final int batchNrToPlot;

    public PlotDecodedOutput(Plot<Double, Double> plot, String outputName, int batchNrToPlot) {
        this(new SpiralPlot(plot), outputName, batchNrToPlot);
    }

    public PlotDecodedOutput(SpiralPlot plot, String outputName, int batchNrToPlot) {
        this.plot = plot;
        this.outputName = outputName;
        this.batchNrToPlot = batchNrToPlot;
        this.plotLabel = outputName + " " + batchNrToPlot;
        this.plot.createSeries(plotLabel);
    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        final INDArray toPlot = activations.get(outputName);
        plot.plot(plotLabel, toPlot, batchNrToPlot);
    }
}
