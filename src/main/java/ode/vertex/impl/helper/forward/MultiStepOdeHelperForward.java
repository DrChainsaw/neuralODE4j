package ode.vertex.impl.helper.forward;

import com.google.common.primitives.Longs;
import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.impl.MultiStepSolver;
import ode.vertex.impl.ForwardPass;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * {@link OdeHelperForward} capable of handling multiple time steps
 *
 * @author Christian Skarby
 */
public class MultiStepOdeHelperForward implements OdeHelperForward {

    private final FirstOrderSolver solver;
    private final INDArray time;

    public MultiStepOdeHelperForward(FirstOrderSolver solver, INDArray time) {
        this.solver = new MultiStepSolver(solver);
        this.time = time;
        if(time.length() <= 2 || !time.isVector()) {
            throw new IllegalArgumentException("time must be a vector! Was of shape: " + Arrays.toString(time.shape())+ "!");
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

        final INDArray z0 = inputs[0].dup();
        final INDArray zt = Nd4j.createUninitialized(Longs.concat(new long[]{time.length() - 1}, z0.shape()));
        solver.integrate(equation, time, inputs[0], zt);

        return alignOutShape(zt, z0);
    }


    private INDArray alignOutShape(INDArray zt, INDArray z0) {
        final long[] shape = zt.shape();
        switch (shape.length) {
            case 3: // Assume recurrent output
                return Nd4j.concat(0, z0.reshape(1, shape[1], shape[2]), zt).permute(1, 2, 0);
            case 5: // Assume conv 3D output
                return Nd4j.concat(0, z0.reshape(1, shape[1], shape[2], shape[3], shape[4]), zt).permute(1, 0, 2, 3, 4);
            // Should not happen as conf throws exception for other types
            default:
                throw new UnsupportedOperationException("Rank not supported: " + zt.rank());
        }
    }
}
