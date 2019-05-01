package util.preproc;

import junit.framework.TestCase;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link DuplicateScalarToShape}
 *
 * @author Christian Skarby
 */
public class DuplicateScalarToShapeTest {

    /**
     * Test duplication to given mini batch size
     */
    @Test
    public void preProcess() {
        final INDArray input = Nd4j.scalar(666);
        final InputPreProcessor preProcessor = new DuplicateScalarToShape();

        assertEquals("Incorrect output!",
                Nd4j.createUninitialized(13).assign(input.sumNumber()),
                preProcessor.preProcess(input, 13, LayerWorkspaceMgr.noWorkspacesImmutable()));
    }

    /**
     * Test duplication to shape [13, 2, 3, 4].
     */
    @Test
    public void preProcess2x3x4() {
        final INDArray input = Nd4j.scalar(666);
        final InputPreProcessor preProcessor = new DuplicateScalarToShape(new long[]{-1, 2, 3, 4});

        assertEquals("Incorrect output!",
                Nd4j.createUninitialized(new long[]{13, 2, 3, 4}).assign(input.sumNumber()),
                preProcessor.preProcess(input, 13, LayerWorkspaceMgr.noWorkspacesImmutable()));
    }

    /**
     * Test back propagation
     */
    @Test
    public void backprop() {
        final INDArray input = Nd4j.arange(2 * 3 * 4).reshape(2, 3, 4);
        assertEquals("Incorrect output!",
                input.sum(),
                new DuplicateScalarToShape().backprop(input, 2, LayerWorkspaceMgr.noWorkspacesImmutable()));
    }

    /**
     * Test that output type is correct
     */
    @Test
    public void getOutputType() {
        final InputType expected = InputType.convolutional(3,4,5);
        final InputPreProcessor preProcessor = new DuplicateScalarToShape(expected.getShape(true));
        final InputType actual = preProcessor.getOutputType(InputType.feedForward(1));
        assertEquals("Incorrect output type!", expected, actual);
    }

    /**
     * Test that {@link DuplicateScalarToShape} can be serialized and deserialized
     * @throws IOException
     */
    @Test
    public void serializeDeserialize() throws IOException {
        final InputPreProcessor preProcessor = new DuplicateScalarToShape(new long[] {666, 13});

        final String json = NeuralNetConfiguration.mapper().writeValueAsString(preProcessor);
        final InputPreProcessor newPreProc = NeuralNetConfiguration.mapper().readValue(json, DuplicateScalarToShape.class);
        TestCase.assertEquals("Not same!", preProcessor, newPreProc);
        TestCase.assertEquals("Not same!", preProcessor.hashCode(), newPreProc.hashCode());
    }
}