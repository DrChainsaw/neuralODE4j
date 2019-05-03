package examples.anode;

import ode.solve.api.StepListener;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.util.List;
import java.util.Map;

/**
 * ResNet model for ANODE experiments
 *
 * @author Christian Skarby
 */
public class ResNetModel implements Model {

    private final ComputationGraph graph;
    private final List<String> resblocks;

    public ResNetModel(ComputationGraph graph, List<String> resblocks) {
        this.graph = graph;
        this.resblocks = resblocks;
    }

    @Override
    public ComputationGraph graph() {
        return graph;
    }

    @Override
    public void plotFeatures(DataSet dataSet, Plot3D plot) {
        final StepListener plotAct = new PlotSteps3D(plot, dataSet.getLabels());

        Map<String, INDArray> acts = graph.feedForward(dataSet.getFeatures(), false);
        for(String resblock: resblocks) {
            plotAct.begin(null, acts.get(resblock));
        }
        // Give plotting a chance to keep up
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    @Override
    public String name() {
        return "resnet";
    }
}
