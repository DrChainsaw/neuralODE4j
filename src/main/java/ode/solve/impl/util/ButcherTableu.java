package ode.solve.impl.util;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Struct for representing the Butcher Tableu.
 *
 * @author Christian Skarby
 */
public class ButcherTableu {

    public final INDArray[] a;
    public final INDArray b;
    public final INDArray bStar;
    public final INDArray c;

    public ButcherTableu(INDArray[] a, INDArray b, INDArray bStar, INDArray c) {
        this.a = a;
        this.b = b;
        this.bStar = bStar;
        this.c = c;
    }
}
