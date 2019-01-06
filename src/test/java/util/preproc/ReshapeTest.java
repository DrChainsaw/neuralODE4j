package util.preproc;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link Reshape}
 *
 * @author Christian Skarby
 */
public class ReshapeTest {

    /**
     * Test reshaping of input
     */
    @Test
    public void preProcess() {
        final INDArray toReshape = Nd4j.arange(3 * 5 * 7 * 11).reshape(3, 5 * 7 * 11);
        final DataSetPreProcessor test = new Reshape(InputType.convolutional(7, 11, 5));

        final DataSet dataSet = new DataSet(toReshape, null);
        test.preProcess(dataSet);

        final long[] expected = {3, 5, 7, 11};
        assertArrayEquals("Incorrect shape!", expected, dataSet.getFeatures().shape());
    }
}