package ode.solve.commons;

import ode.solve.api.FirstOrderEquation;
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Test class for ODEs. Copied from http://commons.apache.org/proper/commons-math/userguide/ode.html
 *
 * @author Christian Skarby
 */
class CircleODE implements FirstOrderEquation, FirstOrderDifferentialEquations {

    private double[] c;
    private double omega;

    CircleODE(double[] c, double omega) {
        this.c     = c;
        this.omega = omega;
    }


    @Override
    public int getDimension() {
        return 2;
    }

    @Override
    public void computeDerivatives(double t, double[] y, double[] yDot) {
        yDot[0] = omega * (c[1] - y[1]);
        yDot[1] = omega * (y[0] - c[0]);
    }

    @Override
    public INDArray calculateDerivative(INDArray y, INDArray t, INDArray fy) {
        fy.putScalar(0, omega * (c[1] - y.getDouble(1)));
        fy.putScalar(1, omega * (y.getDouble(0) - c[0]));
        return fy;
    }
}
