package examples.anode;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

/**
 * Plots the output of each residual block
 *
 * @author Christian Skarby
 */
public class PlotActs extends BaseTrainingListener {

    private final PlotState plot;
    private final List<String> vertexNames;

    public PlotActs(PlotState plot, List<String> vertexNames) {
        this.plot = plot;
        this.vertexNames = vertexNames;
    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {

    }
}
