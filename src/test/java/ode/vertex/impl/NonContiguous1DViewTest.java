package ode.vertex.impl;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
        view.addView(toView, 0, 3);
        view.addView(toView, 6,9);
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
        view.addView(toView, 0, 3);
        view.addView(toView, 6,9);

        final INDArray actual = Nd4j.create(view.length());
        view.assignTo(actual);

        final INDArray expected = Nd4j.create(new double[] {0,1,2,6,7,8});
        assertEquals("Viewed array was not changed!", expected, actual);
    }
}