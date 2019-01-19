package ode.vertex.impl.helper.backward;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.vertex.impl.NonContiguous1DView;
import ode.vertex.impl.helper.NDArrayIndexAccumulator;
import ode.vertex.impl.helper.forward.ForwardPass;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

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
    private final int timeIndex;

    public SingleStepAdjoint(FirstOrderSolver solver, INDArray time, int timeIndex) {
        this.solver = solver;
        this.time = time;
        this.timeIndex = timeIndex;
        if (time.length() != 2 && time.rank() != 1) {
            throw new IllegalArgumentException("time must be a vector with two elements! Was of shape: " + Arrays.toString(time.shape()) + "!");
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> solve(ComputationGraph graph, InputArrays input, MiscPar miscPars) {

        // Create augmented dynamics for adjoint method
        // Initialization: S0:
        // z(t1) = lastoutput
        // a(t) = -dL/d(z(t1)) = -epsilon from next layer (i.e getEpsilon). Use last row if more than one timestep
        // parameters = zeros
        // dL/dt1 = -dL / dz(t1) dot dz(t1) / dt1

        final INDArray dL_dzt1 = input.getLossGradient();
        final INDArray zt1 = input.getLastOutput();
        final NonContiguous1DView realParamGrads = input.getRealGradientView();

        final FirstOrderEquation forward = new ForwardPass(graph,
                miscPars.getWsMgr(),
                false,
                input.getLastInputs());

        final INDArray dzt1_dt1 = forward.calculateDerivative(zt1, time.getColumn(1), zt1.dup());

        final INDArray dL_dt1 = dL_dzt1.reshape(1, dzt1_dt1.length())
                .mmul(dzt1_dt1.reshape(dzt1_dt1.length(), 1)).muli(-1);

        final INDArray zAug = Nd4j.create(1, zt1.length() + dL_dzt1.length() + graph.numParams() + dL_dt1.length());

        final NDArrayIndexAccumulator accumulator = new NDArrayIndexAccumulator(zAug);
        accumulator.increment(zt1.reshape(new long[]{1, zt1.length()}))
                .increment(dL_dzt1.reshape(new long[]{1, dL_dzt1.length()}))
                .increment(Nd4j.zeros(realParamGrads.length()).reshape(new long[]{1, Nd4j.zeros(realParamGrads.length()).length()}))
                .increment(dL_dt1.reshape(new long[]{1, dL_dt1.length()}));

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
        final Gradient gradient = new DefaultGradient(graph.getFlattenedGradients());
        gradient.setGradientFor(miscPars.getGradientParName(), graph.getFlattenedGradients());

        return new Pair<>(gradient, epsilons(miscPars.getWsMgr(), augmentedDynamics, dL_dt1, input.getLastInputs()));
    }

    private INDArray[] epsilons(
            LayerWorkspaceMgr wsMgr,
            AugmentedDynamics finalState,
            INDArray dL_dt1,
            INDArray[] inputs) {
        if (inputs.length != 1) {
            throw new UnsupportedOperationException("More than one inputs not supported! Was: " + inputs.length + "!");
        }
        if(timeIndex != -1) {
            return epsilonsWithTime(wsMgr, finalState, dL_dt1, inputs);
        }
        return epsilonsWithoutTime(wsMgr, finalState, inputs);
    }

    private INDArray[] epsilonsWithTime(
            LayerWorkspaceMgr wsMgr,
            AugmentedDynamics finalState,
            INDArray dL_dt1,
            INDArray[] inputs) {
        final INDArray[] epsilons = new INDArray[inputs.length + 1];
        for (int i = 0; i < inputs.length; i++) {
            if (i != timeIndex) {
                epsilons[i] = wsMgr.leverageTo(ArrayType.ACTIVATION_GRAD, finalState.zAdjoint());
            }
        }
        epsilons[timeIndex] = Nd4j.hstack(dL_dt1, finalState.tAdjoint());
        return epsilons;
    }

    private INDArray[] epsilonsWithoutTime(LayerWorkspaceMgr wsMgr, AugmentedDynamics finalState, INDArray[] inputs) {
        final INDArray[] epsilons = new INDArray[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            epsilons[i] = wsMgr.leverageTo(ArrayType.ACTIVATION_GRAD, finalState.zAdjoint());
        }
        return epsilons;
    }
}
