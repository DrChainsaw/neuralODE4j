package ode.solve.impl.util;

import ode.solve.CircleODE;
import org.apache.commons.math3.ode.ExpandableStatefulODE;
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations;
import org.apache.commons.math3.ode.nonstiff.DormandPrince54Integrator;
import org.apache.commons.math3.util.FastMath;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link AdaptiveRungeKuttaStepPolicy}
 *
 * @author Christian Skarby
 */
public class AdaptiveRungeKuttaStepPolicyTest {

    /**
     * Test step initialization versus reference implementation in {@link DormandPrince54Integrator}.
     */
    @Test
    public void initializeStep() {
        final INDArray t = Nd4j.create(new double[]{-1.2, 7.4});
        final INDArray y0 = Nd4j.create(new double[]{3, 5});
        final CircleODE equation = new CircleODE(new double[]{1.23, 4.56}, 0.666);

        final double absTol = 1.23e-3;
        final double relTol = 2.34e-5;
        final DormandPrince54Integrator reference = new DormandPrince54Integrator(1e-20, 1e20, absTol, relTol) {

            DormandPrince54Integrator setEquation(FirstOrderDifferentialEquations equations) {
                setEquations(new ExpandableStatefulODE(equations));
                return this;
            }
        }.setEquation(equation);

        final double[] y1 = y0.toDoubleVector();
        final double[] yDot = y0.toDoubleVector();
        equation.computeDerivatives(t.getDouble(0), y1, yDot);
        final double[] yDot1 = y0.toDoubleVector();

        final double[] scale = new double[equation.getDimension()];
        for (int i = 0; i < scale.length; ++i) {
            scale[i] = absTol + relTol * FastMath.abs(y0.getDouble(i));
        }

        final double stepRef = reference.initializeStep(
                true, 5, scale, t.getDouble(0), y0.toDoubleVector(), yDot,
                y1, yDot1);

        final FirstOrderEquationWithState eqState = new FirstOrderEquationWithState(equation, t.getColumn(0), y0, 5);
        final INDArray stepAct = new AdaptiveRungeKuttaStepPolicy(
                new SolverConfig(absTol, relTol, 1e-20, 1e20), 5)
                .initializeStep(eqState, t);

        assertEquals("Incorrect step size!", stepRef, stepAct.getDouble(0), 1e-6);
        assertArrayEquals("Incorrect yDot[0]!", yDot, eqState.getStateDot(0).toDoubleVector(), 1e-6);
        assertArrayEquals("Incorrect yDot[1]!", yDot1, eqState.getStateDot(1).toDoubleVector(), 1e-6);
    }

    /**
     * Test step filtering versus reference implementation in {@link DormandPrince54Integrator}
     */
    @Test
    public void filterStep() {
        final INDArray step = Nd4j.create(1).assign(1.23);
        final INDArray error = Nd4j.create(1).assign(0.666);

        final double expected = new DormandPrince54Integrator(1e-20, 1e20, 1e-20, 1e20) {

            double calcError() {
                // Copy pasted from EmbeddedRungeKuttaIntegrator
                final double factor =
                        FastMath.min(getMaxGrowth(),
                                FastMath.max(getMinReduction(), getSafety() * FastMath.pow(error.getDouble(0), -1.0 / getOrder())));
                return filterStep(step.getDouble(0) * factor, true, true);
            }
        }.calcError();

        final INDArray actual = new AdaptiveRungeKuttaStepPolicy(
                new SolverConfig(1e-20, 1e-20, 1e-20, 1e20),
                5)
                .stepForward(step, error);

        assertEquals("Incorrect filtered step!", expected, actual.getDouble(0), 1e-6);
        assertEquals("Step shall not change!", 1.23, step.getDouble(0), 1e-6);
        assertEquals("Error shall not change!", 0.666, error.getDouble(0), 1e-6);
    }
}