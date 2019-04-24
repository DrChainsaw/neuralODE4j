package ode.vertex.impl.helper.forward;

import com.google.common.primitives.Longs;
import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderMultiStepSolver;
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
public class MultiStep implements OdeHelperForward {

    private final FirstOrderMultiStepSolver solver;
    private final INDArray time;

    public MultiStep(FirstOrderMultiStepSolver solver, INDArray time) {
        this.solver = solver;
        this.time = time;
        if(time.length() <= 2 || !time.isVector()) {
            throw new IllegalArgumentException("time must be a vector of size > 2! Was of shape: " + Arrays.toString(time.shape())+ "!");
        }
    }

    @Override
    public INDArray solve(ComputationGraph graph, LayerWorkspaceMgr wsMgr, GraphInput input) {

        final FirstOrderEquation equation = new ForwardPass(
                graph,
                wsMgr,
                true,
                input
        );

        final INDArray y0 = input.y0().dup();
        final INDArray yt = Nd4j.createUninitialized(Longs.concat(new long[]{time.length() - 1}, y0.shape()));
        solver.integrate(equation, time, input.y0().dup(), yt);

        return alignOutShape(yt, y0);
    }


    private INDArray alignOutShape(INDArray yt, INDArray y0) {
        final long[] shape = yt.shape();
        switch (shape.length) {
            case 3: // Assume recurrent output
                return Nd4j.concat(0, y0.reshape(1, shape[1], shape[2]), yt).permute(1, 2, 0);
            case 5: // Assume conv 3D output
                return Nd4j.concat(0, y0.reshape(1, shape[1], shape[2], shape[3], shape[4]), yt).permute(1, 0, 2, 3, 4);
            // Should not happen as conf throws exception for other types
            default:
                throw new UnsupportedOperationException("Rank not supported: " + yt.rank());
        }
    }
}
