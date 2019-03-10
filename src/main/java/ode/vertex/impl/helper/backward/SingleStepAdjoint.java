package ode.vertex.impl.helper.backward;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.vertex.impl.gradview.INDArray1DView;
import ode.vertex.impl.helper.NDArrayIndexAccumulator;
import ode.vertex.impl.helper.forward.ForwardPass;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
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

        if(time.getDouble(0) > time.getDouble(1)) {
            throw new IllegalArgumentException("Time must be in increasing order! Got: " + time);
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
        // Not sure why this is not same as the above. Pytorch treats them differently in original repo
        final INDArray dL_dzt1_time = input.getLossGradientTime();
        final INDArray zt1 = input.getLastOutput();
        final INDArray1DView realParamGrads = input.getRealGradientView();

        final FirstOrderEquation forward = new ForwardPass(graph,
                miscPars.getWsMgr(),
                true, // Always use training as batch norm running mean and var become messed up otherwise. Same effect seen in original pytorch repo.
                input.getLastInputs());

        // TODO: This is only used for computing time gradients. Make it so that it only happens when they are needed
        final INDArray dzt1_dt1 = forward.calculateDerivative(zt1, time.getColumn(1), zt1.dup());

        final INDArray dL_dt1 = dL_dzt1_time.reshape(1, dzt1_dt1.length())
                .mmul(dzt1_dt1.reshape(dzt1_dt1.length(), 1));

        final INDArray zAug = Nd4j.create(1, zt1.length() + dL_dzt1.length() + graph.numParams() + dL_dt1.length());
        final INDArray paramAdj = Nd4j.zeros(realParamGrads.length());
        realParamGrads.assignTo(paramAdj);

        final NDArrayIndexAccumulator accumulator = new NDArrayIndexAccumulator(zAug);
        accumulator.increment(zt1.reshape(new long[]{1, zt1.length()}))
                .increment(dL_dzt1.reshape(new long[]{1, dL_dzt1.length()}))
                .increment(paramAdj.reshape(new long[]{1, paramAdj.length()}))
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

        return new Pair<>(gradient, epsilons(augmentedDynamics, dL_dt1, input.getLastInputs()));
    }

    private INDArray[] epsilons(
            AugmentedDynamics finalState,
            INDArray dL_dt1,
            INDArray[] inputs) {
        if (inputs.length != 1) {
            throw new UnsupportedOperationException("More than one inputs not supported! Was: " + inputs.length + "!");
        }
        if(timeIndex != -1) {
            return epsilonsWithTime(finalState, dL_dt1, inputs);
        }
        return epsilonsWithoutTime(finalState, inputs);
    }

    private INDArray[] epsilonsWithTime(
            AugmentedDynamics finalState,
            INDArray dL_dt1,
            INDArray[] inputs) {
        final INDArray[] epsilons = new INDArray[inputs.length + 1];
        for (int i = 0; i < inputs.length; i++) {
            if (i != timeIndex) {
                epsilons[i] = finalState.zAdjoint();
            }
        }
        epsilons[timeIndex] = Nd4j.hstack(dL_dt1, finalState.tAdjoint());
        return epsilons;
    }

    private INDArray[] epsilonsWithoutTime(AugmentedDynamics finalState, INDArray[] inputs) {
        final INDArray[] epsilons = new INDArray[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            epsilons[i] = finalState.zAdjoint();
        }
        return epsilons;
    }
}
