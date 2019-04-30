package examples.anode;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.DataSet;
import util.plot.Plot;

/**
 * Models for ANODE experiments. In addition to the actual {@link ComputationGraph}, it also has methods to observe the
 * flows of the ODE
 *
 * @author Christian Skarby
 */
public interface Model {

    /**
     * {@link ComputationGraph} for the model
     * @return {@link ComputationGraph} for the model
     */
    ComputationGraph graph();

    /**
     * Plot the flow for the given data set
     * @param dataSet DataSet to plot flow for
     * @param plot Flow will be drawn in this plot
     */
    void plotFlow(DataSet dataSet, Plot<Double, Double> plot);

}
