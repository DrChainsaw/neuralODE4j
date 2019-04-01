package util.listen.training;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import util.plot.Plot;

import java.util.Map;

/**
 * Plots activations as a function of iteration number
 *
 * @author Christian Skarby
 */
public class PlotActivations extends BaseTrainingListener {

    private final Plot<Integer, Double> plot;
    private final String activationName;
    private final String[] labels;

    public PlotActivations(Plot<Integer, Double> plot, String activationName, String[] labels) {
        this.plot = plot;
        this.activationName = activationName;
        this.labels = labels;
    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        final int iteration = ((ComputationGraph)model).getIterationCount();

        final INDArray toPlot = activations.get(activationName).mean(0);

        int labelSwitch = (int)toPlot.length() / labels.length;
        for(int i = 0; i < toPlot.length(); i++) {
            plot.plotData( labels[i / labelSwitch] + "_" + (i % labelSwitch), iteration, toPlot.getDouble(i));
        }
    }
}
