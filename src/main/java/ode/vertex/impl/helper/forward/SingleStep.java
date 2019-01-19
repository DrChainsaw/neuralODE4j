package ode.vertex.impl.helper.forward;


import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.vertex.impl.ForwardPass;
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
    public INDArray solve(ComputationGraph graph, LayerWorkspaceMgr wsMgr, INDArray[] inputs) {
        if (inputs.length != 1) {
            throw new IllegalArgumentException("Only single input supported!");
        }

        final FirstOrderEquation equation = new ForwardPass(
                graph,
                wsMgr,
                true,
                inputs
        );

        final INDArray z0 = inputs[0];
        final INDArray zt = Nd4j.createUninitialized(z0.shape());
        solver.integrate(equation, time, z0, zt);

        return zt;
    }
}
