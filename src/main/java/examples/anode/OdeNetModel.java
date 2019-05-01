package examples.anode;

import ode.solve.api.FirstOrderSolverConf;
import ode.solve.api.StepListener;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.DataSet;
import util.plot.Plot;

/**
 * ODE model for ANODE experiments
 *
 * @author Christian Skarby
 */
public class OdeNetModel implements Model {

    private final ComputationGraph graph;
    private final FirstOrderSolverConf solverConf;
    private final String name;

    public OdeNetModel(ComputationGraph graph, FirstOrderSolverConf solverConf, String name) {
        this.graph = graph;
        this.solverConf = solverConf;
        this.name = name;
    }

    @Override
    public ComputationGraph graph() {
        return graph;
    }

    @Override
    public void plotFeatures(DataSet dataSet, Plot<Double, Double> plot) {
        plot.clearData();

        final StepListener listener = new PlotState(plot, dataSet.getLabels());
        solverConf.addListeners(listener);

        graph.output(dataSet.getFeatures());

        solverConf.clearListeners(listener);
    }

    @Override
    public String name() {
        return name;
    }
}
