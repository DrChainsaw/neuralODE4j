package ode.vertex.impl.gradview;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link Contiguous1DView}
 *
 * @author Christian Skarby
 */
public class Contiguous1DViewTest {

    /**
     * Test assignment of {@link Contiguous1DView} from another {@link INDArray}. Set first and last three elements
     * and leave three in the middle untouched
     */
    @Test
    public void assignFrom() {
        final INDArray toView = Nd4j.ones(13);
        final INDArray other = Nd4j.zeros(toView.shape()).reshape(toView.length());

        final INDArray1DView view = new Contiguous1DView(toView);

        view.assignFrom(other);
        assertEquals("View not set!", other, toView);
    }

    /**
     * Test assignment to another {@link INDArray} from a {@link Contiguous1DView}.
     */
    @Test
    public void assignTo() {
        final INDArray toView = Nd4j.ones(13);
        final INDArray other = Nd4j.zeros(toView.shape()).reshape(toView.length());

        final INDArray1DView view = new Contiguous1DView(toView);

        view.assignTo(other);
        assertEquals("View not set!", other, toView);
    }

    /**
     * Test that length is correct
     */
    @Test
    public void length() {
        final long length = 27;

        assertEquals("Incorrect length!", length, new Contiguous1DView(Nd4j.ones(length)).length());
    }
}