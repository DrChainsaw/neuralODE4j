package ode.vertex.impl.helper.backward;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.vertex.impl.gradview.INDArray1DView;
import ode.vertex.impl.helper.NDArrayIndexAccumulator;
import ode.vertex.impl.helper.backward.timegrad.TimeGrad;
import ode.vertex.impl.helper.forward.ForwardPass;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * {@link OdeHelperBackward} using the adjoint method capable of handling a single time step. Gradients for time steps
 * will only be provided if required.
 *
 * @author Christian Skarby
 */
public class SingleStepAdjoint implements OdeHelperBackward {

    private final FirstOrderSolver solver;
    private final INDArray time;
    private final TimeGrad.Factory timeGradFactory;

    public SingleStepAdjoint(FirstOrderSolver solver, INDArray time, TimeGrad.Factory timeGradFactory) {
        this.solver = solver;
        this.time = time;
        this.timeGradFactory = timeGradFactory;
        if (time.length() != 2 && time.rank() != 1) {
            throw new IllegalArgumentException("time must be a vector with two elements! Was of shape: " + Arrays.toString(time.shape()) + "!");
        }
    }

    @Override
    public INDArray[] solve(ComputationGraph graph, InputArrays input, MiscPar miscPars) {

        // Create augmented dynamics for adjoint method
        // Initialization: S0:
        // z(t1) = lastoutput
        // a(t) = -dL/d(z(t1)) = -epsilon from next layer (i.e getEpsilon). Use last row if more than one timestep
        // parameters = zeros
        // dL/dt1 = -dL / dz(t1) dot dz(t1) / dt1

        final INDArray dL_dzt1 = input.getLossGradient();
        final INDArray zt1 = input.getLastOutput();
        final INDArray1DView realParamGrads = input.getRealGradientView();

        final FirstOrderEquation forward = new ForwardPass(graph,
                miscPars.getWsMgr(),
                true, // Always use training as batch norm running mean and var become messed up otherwise. Same effect seen in original pytorch repo.
                input.getLastInputs());

        final TimeGrad timeGrad = timeGradFactory.create();
        final INDArray dL_dt1 = timeGrad.calcTimeGradT1(forward, zt1, time);

        final INDArray zAug = Nd4j.create(1, zt1.length() + dL_dzt1.length() + graph.numParams() + dL_dt1.length());
        final INDArray paramAdj = Nd4j.zeros(realParamGrads.length());
        realParamGrads.assignTo(paramAdj);

        final NDArrayIndexAccumulator accumulator = new NDArrayIndexAccumulator(zAug);
        accumulator.increment(zt1.reshape(new long[]{1, zt1.length()}))
                .increment(dL_dzt1.reshape(new long[]{1, dL_dzt1.length()}))
                .increment(paramAdj.reshape(new long[]{1, paramAdj.length()}))
                .increment(dL_dt1);

        final AugmentedDynamics augmentedDynamics = new AugmentedDynamics(
                zAug,
                dL_dzt1.shape(),
                new long[]{realParamGrads.length()},
                dL_dt1.shape());

        final FirstOrderEquation equation = new BackpropagateAdjoint(
                augmentedDynamics,
                forward,
                new BackpropagateAdjoint.GraphInfo(graph, realParamGrads, miscPars.getWsMgr(), miscPars.isUseTruncatedBackPropTroughTime())
        );

        INDArray augAns = solver.integrate(equation, Nd4j.reverse(time.dup()), zAug, zAug.dup());

        augmentedDynamics.updateFrom(augAns);

        realParamGrads.assignFrom(augmentedDynamics.paramAdjoint());

        return timeGrad.createLossGradient(augmentedDynamics.zAdjoint(), augmentedDynamics.tAdjoint());
    }
}
