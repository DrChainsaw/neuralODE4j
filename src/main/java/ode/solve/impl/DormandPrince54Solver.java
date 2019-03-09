package ode.solve.impl;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.StepListener;
import ode.solve.conf.SolverConfig;
import ode.solve.impl.util.AdaptiveRungeKuttaStepPolicy;
import ode.solve.impl.util.ButcherTableu;
import ode.solve.impl.util.SolverConfigINDArray;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

/**
 * Implementation of the Dormand-Prince method for solving ordinary differential equations.
 * <br><br>
 * Translated implementation from {@link org.apache.commons.math3.ode.nonstiff.DormandPrince54Integrator} to use INDArrays.
 *
 * @author Christian Skarby
 */
public class DormandPrince54Solver implements FirstOrderSolver {

    private static final ButcherTableu.Builder butcherTableuBuilder =
            ButcherTableu.builder()
                    .a(new double[][]{
                            {1.0 / 5.0},
                            {3.0 / 40.0, 9.0 / 40.0},
                            {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0},
                            {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0},
                            {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0},
                            {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0}
                    })
                    .b(new double[]{
                            35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0
                    })
                    .bStar(new double[]{
                            71.0 / 57600.0, 0.0, -71.0 / 16695.0, 71.0 / 1920.0, -17253.0 / 339200.0, 22.0 / 525.0, -1.0 / 40.0
                    })
                    .c(new double[]{
                            1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0
                    })
                    .cMid(new double[]{
                            6025192743d / 30085553152d / 2d, 0, 51252292925d / 65400821598d / 2d, -2691868925d / 45128329728d / 2d,
                            187940372067d / 1594534317056d / 2d, -1776094331 / 19743644256d / 2d, 11237099 / 235043384d / 2d});

    private final AdaptiveRungeKuttaSolver solver;

    /**
     * {@link AdaptiveRungeKuttaSolver.MseComputation} for DormandPrince54
     *
     * @author Christian Skarby
     */
    public static class DormandPrince54Mse implements AdaptiveRungeKuttaSolver.MseComputation {

        private final SolverConfigINDArray config;
        private final INDArray errorCoeffs;

        final static WorkspaceConfiguration wsConf = WorkspaceConfiguration.builder()
                .overallocationLimit(0.0)
                .policySpill(SpillPolicy.REALLOCATE)
                .build();

        public DormandPrince54Mse(SolverConfigINDArray config) {
            this(config, butcherTableuBuilder.build().bStar);
        }

        public DormandPrince54Mse(SolverConfigINDArray config, INDArray errorCoeffs) {
            this.config = config;
            this.errorCoeffs = errorCoeffs;
        }

        @Override
        public INDArray estimateMse(
                final INDArray yDotK,
                final INDArray y0,
                final INDArray y1,
                final INDArray h
        ) {

            // TODO: Test remove zero row from bStar and yDotK and see if there are net gains
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(wsConf, this.getClass().getSimpleName())) {
                final INDArray errSum = errorCoeffs.mmul(yDotK);
                final INDArray yScale = max(abs(y0), abs(y1), false).reshape(new long[]{1, y0.length()});
                final INDArray tol = yScale.muli(config.getRelTol()).addi(config.getAbsTol());
                final INDArray ratio = errSum.divi(tol).muli(h);
                final INDArray error = ratio.muli(ratio);
                return sqrt(error.mean()).detach();
            }
        }
    }

    public DormandPrince54Solver(SolverConfig config) {
        final SolverConfigINDArray configINDArray = new SolverConfigINDArray(config);
        solver = new AdaptiveRungeKuttaSolver(
                butcherTableuBuilder.build(),
                new AdaptiveRungeKuttaStepPolicy(configINDArray, 5),
                new DormandPrince54Mse(configINDArray));
    }

    @Override
    public INDArray integrate(FirstOrderEquation equation, INDArray t, INDArray y0, INDArray yOut) {
        INDArray ret = solver.integrate(equation, t, y0, yOut);
        return ret;
    }

    @Override
    public void addListener(StepListener... listeners) {
        solver.addListener(listeners);
    }

    @Override
    public void clearListeners(StepListener... listeners) {
        solver.clearListeners(listeners);
    }
}
