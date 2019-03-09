package ode.vertex.impl;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link NonContiguous1DView}
 *
 * @author Christian Skarby
 */
public class NonContiguous1DViewTest {

    /**
     * Test assignment of {@link NonContiguous1DView} from another {@link INDArray}. Set first and last three elements
     * and leave three in the middle untouched
     */
    @Test
    public void assignFrom() {
        final INDArray toView = Nd4j.zeros(9);

        final NonContiguous1DView view = new NonContiguous1DView();
        view.addView(toView.get(NDArrayIndex.interval(0, 3)));
        view.addView(toView.get(NDArrayIndex.interval(6, 9)));
        view.assignFrom(Nd4j.ones(new long[] {6}));

        final INDArray expected = Nd4j.create(new double[] {1,1,1,0,0,0,1,1,1});
        assertEquals("Viewed array was not changed!", expected, toView);
    }

    /**
     * Test assignment to another {@link INDArray} from a {@link NonContiguous1DView}.
     */
    @Test
    public void assignTo() {
        final INDArray toView = Nd4j.linspace(0,8,9);

        final NonContiguous1DView view = new NonContiguous1DView();
        view.addView(toView.get(NDArrayIndex.interval(0, 3)));
        view.addView(toView.get(NDArrayIndex.interval(6, 9)));

        final INDArray actual = Nd4j.create(view.length());
        view.assignTo(actual);

        final INDArray expected = Nd4j.create(new double[] {0,1,2,6,7,8});
        assertEquals("Viewed array was not changed!", expected, actual);
    }

    /**
     * Test assignment of {@link NonContiguous1DView} with one 2x3x4 view and one 2x2 view from another {@link INDArray}.
     */
    @Test
    public void assignFrom2x3x4and2x2() {
        final INDArray toView = Nd4j.zeros(2*3*4*5);

        final NonContiguous1DView view = new NonContiguous1DView();
        view.addView(toView.get(NDArrayIndex.interval(0, 2*3*4)).reshape(2,3,4));
        view.addView(toView.get(NDArrayIndex.interval(2*3*4+4, 2*3*4+8)).reshape(2,2));
        view.assignFrom(Nd4j.ones(view.length()));

        final double expectedSum = view.length();
        assertEquals("Viewed array was not changed!", expectedSum, toView.sumNumber().doubleValue(),1e-10);
    }

    /**
     * Test assignment to another {@link INDArray} from a {@link NonContiguous1DView}.
     */
    @Test
    public void assignTo2x3x4() {
        final INDArray toView = Nd4j.linspace(0,36,37);

        final NonContiguous1DView view = new NonContiguous1DView();
        view.addView(toView.get(NDArrayIndex.interval(0, 12)).reshape(2,3,2));
        view.addView(toView.get(NDArrayIndex.interval(17, 17+10)).reshape(5,2));

        final INDArray actual = Nd4j.create(view.length());
        view.assignTo(actual);

        final INDArray expected = Nd4j.create(new double[] {0,1,2,3,4,5,6,7,8,9,10,11, 17,18,19,20,21,22,23,24,25,26});
        assertEquals("Viewed array was not changed!", expected, actual);
    }
}