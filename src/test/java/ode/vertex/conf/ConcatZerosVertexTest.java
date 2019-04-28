package ode.vertex.conf;

import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotSame;

/**
 * Test cases for {@link ConcatZerosVertex}
 *
 * @author Christian Skarby
 */
public class ConcatZerosVertexTest {

    /**
     * Test cloning
     */
    @Test
    public void cloneIt() {
        final GraphVertex vertex = new ConcatZerosVertex(13);
        final GraphVertex clone = vertex.clone();
        assertNotSame("Must not be same instance!", vertex, clone);
        assertEquals("Must be equal!", vertex, clone);
    }

    /**
     * Test output types
     */
    @Test
    public void getOutputType() {
        final GraphVertex vertex = new ConcatZerosVertex(7);
        assertEquals("Incorrect output type!",
                vertex.getOutputType(-1, InputType.feedForward(3)),
                InputType.feedForward(10));

        assertEquals("Incorrect output type!", vertex.getOutputType(-1,
                InputType.recurrent(5, 10)),
                InputType.recurrent(12, 10));

        assertEquals("Incorrect output type!", vertex.getOutputType(-1,
                InputType.convolutional(2, 3, 4)),
                InputType.convolutional(2, 3, 11));


        assertEquals("Incorrect output type!", vertex.getOutputType(-1,
                InputType.convolutionalFlat(2, 3, 4)),
                InputType.convolutionalFlat(2, 3, 11));

        assertEquals("Incorrect output type!", vertex.getOutputType(-1,
                InputType.convolutional3D(Convolution3D.DataFormat.NDHWC, 2, 3, 4, 5)),
                InputType.convolutional3D(Convolution3D.DataFormat.NDHWC, 2, 3, 4, 12));
    }


    /**
     * Test memory report to string method
     */
    @Test
    public void getMemoryReport() {
        assertEquals("Incorrect memory report!",
                "LayerMemoryReport(layerName=null,layerType=ConcatZerosVertex)",
                new ConcatZerosVertex(7).getMemoryReport(InputType.feedForward(3)).toString());
    }

    /**
     * Test that two {@link ConcatZerosVertex}es with same number of zeros are equal
     */
    @Test
    public void equalToSame() {
        assertEquals("Shall be equal!", new ConcatZerosVertex(5), new ConcatZerosVertex(5));
    }

    /**
     * Test that two {@link ConcatZerosVertex}es with different number of zeros are not equal
     */
    @Test
    public void notEqualToDifferent() {
        assertNotEquals("Shall not be equal!", new ConcatZerosVertex(5), new ConcatZerosVertex(4));
    }

    /**
     * Test that two {@link ConcatZerosVertex}es with same number of zeros have the same hashcode
     */
    @Test
    public void hashCodeSame() {
        assertEquals("Shall be equal!", new ConcatZerosVertex(5).hashCode(), new ConcatZerosVertex(5).hashCode());
    }

    /**
     * Test that two {@link ConcatZerosVertex}es with different number of zeros have different hashcodes
     */
    @Test
    public void hashCodeDifferent() {
        assertNotEquals("Shall not be equal!", new ConcatZerosVertex(5).hashCode(), new ConcatZerosVertex(4).hashCode());
    }
}