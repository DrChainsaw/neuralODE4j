package util.preproc;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotSame;

/**
 * Test cases for {@link ConcatZeros}
 */
public class ConcatZerosTest {

    /**
     * Test that the correct number of zeroes are appended
     */
    @Test
    public void preProcess2D() {
        final long nrofZeros = 5;
        final InputPreProcessor vertex = new ConcatZeros(nrofZeros);

        final INDArray input = Nd4j.ones(2, 3);
        final INDArray expected = createAppended(input, nrofZeros, 0);
        assertEquals("Incorrect output!", expected, vertex.preProcess(input, (int) input.size(0), LayerWorkspaceMgr.noWorkspacesImmutable()));

    }

    /**
     * Test that the correct number of zeroes are appended
     */
    @Test
    public void preProcess3D() {
        final long nrofZeros = 3;
        final InputPreProcessor vertex = new ConcatZeros(nrofZeros);


        final INDArray input = Nd4j.ones(2, 3, 4);
        final INDArray expected = createAppended(input, nrofZeros, 0);

        assertEquals("Incorrect output!", expected, vertex.preProcess(input, (int) input.size(0), LayerWorkspaceMgr.noWorkspacesImmutable()));
    }

    /**
     * Test that the correct number of zeroes are appended
     */
    @Test
    public void preProcess4D() {
        final long nrofZeros = 7;
        final InputPreProcessor vertex = new ConcatZeros(nrofZeros);


        final INDArray input = Nd4j.ones(2, 3, 4, 5);
        final INDArray expected = createAppended(input, nrofZeros, 0);

        assertEquals("Incorrect output!", expected, vertex.preProcess(input, (int) input.size(0), LayerWorkspaceMgr.noWorkspacesImmutable()));
    }

    /**
     * Test that the correct number of zeroes are appended
     */
    @Test
    public void preProcess5D() {
        final long nrofZeros = 2;
        final InputPreProcessor vertex = new ConcatZeros(nrofZeros);


        final INDArray input = Nd4j.ones(2, 3, 4, 5, 6);
        final INDArray expected = createAppended(input, nrofZeros, 0);

        assertEquals("Incorrect output!", expected, vertex.preProcess(input, (int) input.size(0), LayerWorkspaceMgr.noWorkspacesImmutable()));
    }

    /**
     * Test that zeroes are removed from backwards pass
     */
    @Test
    public void doBackward2D() {
        final long nrofZeros = 5;
        final InputPreProcessor vertex = new ConcatZeros(nrofZeros);


        final INDArray expected = Nd4j.ones(2, 3);
        final INDArray epsilon = createAppended(expected, nrofZeros, 666);

        assertEquals("Incorrect output!", expected,
                vertex.backprop(epsilon, (int) epsilon.size(0), LayerWorkspaceMgr.noWorkspacesImmutable()));

    }

    /**
     * Test that zeroes are removed from backwards pass
     */
    @Test
    public void doBackward3D() {
        final long nrofZeros = 3;
        final InputPreProcessor vertex = new ConcatZeros(nrofZeros);


        final INDArray expected = Nd4j.ones(2, 3, 4);
        final INDArray epsilon = createAppended(expected, nrofZeros, 666);

        assertEquals("Incorrect output!", expected,
                vertex.backprop(epsilon, (int) epsilon.size(0), LayerWorkspaceMgr.noWorkspacesImmutable()));
    }

    /**
     * Test that zeroes are removed from backwards pass
     */
    @Test
    public void doBackward4D() {
        final long nrofZeros = 7;
        final InputPreProcessor vertex = new ConcatZeros(nrofZeros);


        final INDArray expected = Nd4j.ones(2, 3, 4, 5);
        final INDArray epsilon = createAppended(expected, nrofZeros, 666);

        assertEquals("Incorrect output!", expected,
                vertex.backprop(epsilon, (int) epsilon.size(0), LayerWorkspaceMgr.noWorkspacesImmutable()));
    }

    /**
     * Test that zeroes are removed from backwards pass
     */
    @Test
    public void doBackward5D() {
        final long nrofZeros = 2;
        final InputPreProcessor vertex = new ConcatZeros(nrofZeros);


        final INDArray expected = Nd4j.ones(2, 3, 4, 5, 6);
        final INDArray epsilon = createAppended(expected, nrofZeros, 666);

        assertEquals("Incorrect output!", expected,
                vertex.backprop(epsilon, (int) epsilon.size(0), LayerWorkspaceMgr.noWorkspacesImmutable()));
    }


    private INDArray createAppended(INDArray input, long nrofZeros, float toAppend) {
        INDArrayIndex[] inds = NDArrayIndex.allFor(input);
        final long[] expectedShape = input.shape().clone();
        inds[1] = NDArrayIndex.interval(0, expectedShape[1]);
        expectedShape[1] += nrofZeros;
        final INDArray appended = Nd4j.createUninitialized(expectedShape).assign(toAppend);
        appended.put(inds, input);
        return appended;
    }

    /**
     * Test cloning
     */
    @Test
    public void cloneIt() {
        final InputPreProcessor vertex = new ConcatZeros(13);
        final InputPreProcessor clone = vertex.clone();
        assertNotSame("Must not be same instance!", vertex, clone);
        assertEquals("Must be equal!", vertex, clone);
    }

    /**
     * Test output types
     */
    @Test
    public void getOutputType() {
        final InputPreProcessor vertex = new ConcatZeros(7);
        assertEquals("Incorrect output type!",
                vertex.getOutputType(InputType.feedForward(3)),
                InputType.feedForward(10));

        assertEquals("Incorrect output type!", vertex.getOutputType(
                InputType.recurrent(5, 10)),
                InputType.recurrent(12, 10));

        assertEquals("Incorrect output type!", vertex.getOutputType(
                InputType.convolutional(2, 3, 4)),
                InputType.convolutional(2, 3, 11));


        assertEquals("Incorrect output type!", vertex.getOutputType(
                InputType.convolutionalFlat(2, 3, 4)),
                InputType.convolutionalFlat(2, 3, 11));

        assertEquals("Incorrect output type!", vertex.getOutputType(
                InputType.convolutional3D(Convolution3D.DataFormat.NDHWC, 2, 3, 4, 5)),
                InputType.convolutional3D(Convolution3D.DataFormat.NDHWC, 2, 3, 4, 12));
    }

    /**
     * Test that two {@link ConcatZeros}es with same number of zeros are equal
     */
    @Test
    public void equalToSame() {
        assertEquals("Shall be equal!", new ConcatZeros(5), new ConcatZeros(5));
    }

    /**
     * Test that two {@link ConcatZeros}es with different number of zeros are not equal
     */
    @Test
    public void notEqualToDifferent() {
        assertNotEquals("Shall not be equal!", new ConcatZeros(5), new ConcatZeros(4));
    }

    /**
     * Test that two {@link ConcatZeros}es with same number of zeros have the same hashcode
     */
    @Test
    public void hashCodeSame() {
        assertEquals("Shall be equal!", new ConcatZeros(5).hashCode(), new ConcatZeros(5).hashCode());
    }

    /**
     * Test that two {@link ConcatZeros}es with different number of zeros have different hashcodes
     */
    @Test
    public void hashCodeDifferent() {
        assertNotEquals("Shall not be equal!", new ConcatZeros(5).hashCode(), new ConcatZeros(4).hashCode());
    }

    /**
     * Test that {@link ConcatZeros} can be serialized and deserialized
     * @throws IOException
     */
    @Test
    public void serializeDeserialize() throws IOException {
        final InputPreProcessor preProcessor = new ConcatZeros(666);

        final String json = NeuralNetConfiguration.mapper().writeValueAsString(preProcessor);
        final InputPreProcessor newPreProc = NeuralNetConfiguration.mapper().readValue(json, ConcatZeros.class);
        assertEquals("Not same!", preProcessor, newPreProc);
        assertEquals("Not same!", preProcessor.hashCode(), newPreProc.hashCode());
    }
}