package util.preproc;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link ShiftDim}
 *
 * @author Christian Skarby
 */
public class ShiftDimTest {

    /**
     * Test to shift a 1D array 2 steps to the left
     */
    @Test
    public void noShift() {
        final INDArray toShift = Nd4j.linspace(1, 6, 6);

        final DataSetPreProcessor shift = new ShiftDim(0, () -> 0);
        shift.preProcess(new DataSet(toShift.dup(), null));
        assertEquals("Incorrect output!", toShift, toShift);
    }

    /**
     * Test to shift a 1D array 2 steps to the left
     */
    @Test
    public void shift1dLeft() {
        final INDArray toShift = Nd4j.linspace(1, 6, 6);
        final INDArray expected = Nd4j.create(new double[]{0, 0, 1, 2, 3, 4});

        final DataSetPreProcessor shift = new ShiftDim(0, () -> 2);
        shift.preProcess(new DataSet(toShift, null));
        assertEquals("Incorrect output!", expected, toShift);
    }

    /**
     * Test to shift a 1D array 2 steps to the right
     */
    @Test
    public void shift1dRight() {
        final INDArray toShift = Nd4j.linspace(1, 6, 6);
        final INDArray expected = Nd4j.create(new double[]{3, 4, 5, 6, 0, 0});

        final DataSetPreProcessor shift = new ShiftDim(0, () -> -2);
        shift.preProcess(new DataSet(toShift, null));
        assertEquals("Incorrect output!", expected, toShift);
    }

    /**
     * Test to shift a 2D array 3 steps to the left
     */
    @Test
    public void shift2dLeft() {
        final INDArray toShift = Nd4j.arange(1,16).reshape(3, 5);
        final INDArray expected = Nd4j.hstack(Nd4j.zeros(3, 3), toShift.getColumns(0, 1));

        final DataSetPreProcessor shift = new ShiftDim(1, () -> 3);
        shift.preProcess(new DataSet(toShift, null));
        assertEquals("Incorrect output!", expected, toShift);
    }

    /**
     * Test to shift a 2D array 3 steps to the right
     */
    @Test
    public void shift2dRight() {
        final INDArray toShift = Nd4j.arange(1,16).reshape(3, 5);
        final INDArray expected = Nd4j.hstack(toShift.getColumns(3, 4), Nd4j.zeros(3, 3));

        final DataSetPreProcessor shift = new ShiftDim(1, () -> -3);
        shift.preProcess(new DataSet(toShift, null));
        assertEquals("Incorrect output!", expected, toShift);
    }

    /**
     * Test to shift a 2D array 1 step down
     */
    @Test
    public void shift2dDown() {
        final INDArray toShift = Nd4j.arange(1,16).reshape(3, 5);
        final INDArray expected = Nd4j.vstack(Nd4j.zeros(1, 5), toShift.getRows(0, 1));

        final DataSetPreProcessor shift = new ShiftDim(0, () -> 1);
        shift.preProcess(new DataSet(toShift, null));
        assertEquals("Incorrect output!", expected, toShift);
    }

    /**
     * Test to shift a 2D array 1 step down
     */
    @Test
    public void shift2dUp() {
        final INDArray toShift = Nd4j.arange(1,16).reshape(3, 5);
        final INDArray expected = Nd4j.vstack(toShift.getRows(1, 2), Nd4j.zeros(1, 5));

        final DataSetPreProcessor shift = new ShiftDim(0, () -> -1);
        shift.preProcess(new DataSet(toShift, null));
        assertEquals("Incorrect output!", expected, toShift);
    }
}