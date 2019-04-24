package ode.vertex.impl.helper.forward;


import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Simple {@link OdeHelperForward} capable of only a single time step. Main difference compared to using
 * {@link MultiStep} is that the latter will return output (zt) which includes the first step (z0)
 * as well.
 */
public class SingleStep implements OdeHelperForward {

    private final FirstOrderSolver solver;
    private final INDArray time;

    public SingleStep(FirstOrderSolver solver, INDArray time) {
        this.solver = solver;
        this.time = time;
        if(time.length() != 2 && time.rank() != 1) {
            throw new IllegalArgumentException("time must be a vector with two elements! Was of shape: " + Arrays.toString(time.shape())+ "!");
        }
    }

    @Override
    public INDArray solve(ComputationGraph graph, LayerWorkspaceMgr wsMgr, GraphInput input) {

        final FirstOrderEquation equation = new ForwardPass(
                graph,
                wsMgr,
                true, // Always use training as batch norm running mean and var become messed up otherwise. Same effect seen in original pytorch repo.
                input
        );

        final INDArray y0 = input.y0();
        final INDArray yt = Nd4j.createUninitialized(y0.shape());
        solver.integrate(equation, time, y0, yt);

        return yt;
    }
}
