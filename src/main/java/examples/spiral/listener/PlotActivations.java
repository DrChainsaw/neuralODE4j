package examples.spiral.listener;

import org.deeplearning4j.nn.api.Model;
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

    private int iteration = 0;

    public PlotActivations(Plot<Integer, Double> plot, String activationName) {
        this.plot = plot;
        this.activationName = activationName;
    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        final INDArray toPlot = activations.get(activationName).mean(0);
        for(int i = 0; i < toPlot.length(); i++) {
            plot.plotData(activationName+"_"+ i, iteration, toPlot.getDouble(i));
        }
        iteration++;
    }

}
