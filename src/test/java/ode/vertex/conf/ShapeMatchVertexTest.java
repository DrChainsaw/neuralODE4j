package ode.vertex.conf;

import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Test cases for {@link ShapeMatchVertex}
 *
 * @author Christian Skarby
 */
public class ShapeMatchVertexTest {

    /**
     * Verify that an exception is thrown if an invalid vertex is used
     */
    @Test(expected = IllegalArgumentException.class)
    public void invalidVertex() {
        new ShapeMatchVertex(new ScaleVertex(666));
    }

    /**
     * Test that clone returns an identical copy but not the same instance
     */
    @Test
    public void cloneIt() {
        final GraphVertex vertex = new ShapeMatchVertex(new MergeVertex());
        final GraphVertex clone = vertex.clone();
        assertNotSame("Must not be same instance!", vertex, clone);
        assertEquals("Must be equal!", vertex, clone);
    }

    /**
     * Test that numParams is correct. Kinda bad test case as I CBA to newPlot a mock vertex with numParams > 0 as there
     * is no such vertex which also takes more than one input.
     */
    @Test
    public void numParams() {
        final GraphVertex expected = new ElementWiseVertex(ElementWiseVertex.Op.Add);

        assertEquals("Not same number of params!",
                expected.numParams(false),
                new ShapeMatchVertex(expected).numParams(false));
    }

    /**
     * Test that minVertexInputs is correct
     */
    @Test
    public void minVertexInputs() {
        final GraphVertex expected = new MergeVertex();

        assertEquals("Not same number of inputs!",
                expected.minVertexInputs(),
                new ShapeMatchVertex(expected).minVertexInputs());
    }

    /**
     * Test that maxVertexInputs is correct
     */
    @Test
    public void maxVertexInputs() {
        final GraphVertex expected = new ElementWiseVertex(ElementWiseVertex.Op.Add);

        assertEquals("Not same number of inputs!",
                expected.maxVertexInputs(),
                new ShapeMatchVertex(expected).maxVertexInputs());
    }

    /**
     * Test that output type is as expected
     */
    @Test
    public void getOutputType() {
        final InputType inputType = InputType.convolutional(3, 4, 5);
        final GraphVertex expected = new MergeVertex();

        final InputType actual = new ShapeMatchVertex(expected)
                .getOutputType(-1, inputType, InputType.feedForward(1));

        assertEquals("Incorrect output type!",
                expected.getOutputType(-1, inputType, inputType),
                actual);

    }

    /**
     * Test that output type is as expected when a merge vertex is used
     */
    @Test
    public void getOutputTypeMergeVertex() {
        final InputType inputType1 = InputType.convolutional(3, 4, 5);
        final InputType inputType2 = InputType.convolutional(3, 4, 1);
        final MergeVertex expected = new MergeVertex();

        // Note, type of "expected" matters!
        final InputType actual = new ShapeMatchVertex(expected)
                .getOutputType(-1, inputType1, InputType.feedForward(1));

        assertEquals("Incorrect output type!",
                expected.getOutputType(-1, inputType1, inputType2),
                actual);

    }

    /**
     * Test memory report
     */
    @Test
    public void getMemoryReport() {
        final InputType inputtype = InputType.recurrent(2,3);
        final GraphVertex expected = new ElementWiseVertex(ElementWiseVertex.Op.Max);

        assertEquals("Incorrect memory report!",
                expected.getMemoryReport(inputtype, inputtype),
                new ShapeMatchVertex(expected).getMemoryReport(inputtype, inputtype));

    }

    /**
     * Test equals
     */
    @Test
    public void equals() {
        assertEquals("Expected same!",
                new ShapeMatchVertex(new ElementWiseVertex(ElementWiseVertex.Op.Add)),
                new ShapeMatchVertex(new ElementWiseVertex(ElementWiseVertex.Op.Add)));

        assertNotEquals("Expected not same!",
                new ShapeMatchVertex(new ElementWiseVertex(ElementWiseVertex.Op.Add)),
                new ShapeMatchVertex(new MergeVertex()));

        assertNotEquals("Expected not same!",
                new ShapeMatchVertex(new MergeVertex()),
                new MergeVertex());
    }
}