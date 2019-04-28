package ode.vertex.impl;

import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link ConcatZerosVertex}
 *
 * @author Christian Skarby
 */
public class ConcatZerosVertexTest {

    /**
     * Test that the correct number of zeroes are appended
     */
    @Test
    public void doForward2D() {
        final long nrofZeros = 5;
        final GraphVertex vertex = new ode.vertex.conf.ConcatZerosVertex(nrofZeros)
                .instantiate(null, "test", 0, null, true);

        final INDArray input = Nd4j.ones(2, 3);
        final INDArray expected = createAppended(input, nrofZeros, 0);
        vertex.setInputs(input);
        assertEquals("Incorrect output!", expected, vertex.doForward(true, LayerWorkspaceMgr.noWorkspacesImmutable()));

    }

    /**
     * Test that the correct number of zeroes are appended
     */
    @Test
    public void doForward3D() {
        final long nrofZeros = 3;
        final GraphVertex vertex = new ode.vertex.conf.ConcatZerosVertex(nrofZeros)
                .instantiate(null, "test", 0, null, true);

        final INDArray input = Nd4j.ones(2, 3, 4);
        final INDArray expected = createAppended(input, nrofZeros, 0);
        vertex.setInputs(input);
        assertEquals("Incorrect output!", expected, vertex.doForward(true, LayerWorkspaceMgr.noWorkspacesImmutable()));
    }

    /**
     * Test that the correct number of zeroes are appended
     */
    @Test
    public void doForward4D() {
        final long nrofZeros = 7;
        final GraphVertex vertex = new ode.vertex.conf.ConcatZerosVertex(nrofZeros)
                .instantiate(null, "test", 0, null, true);

        final INDArray input = Nd4j.ones(2, 3, 4, 5);
        final INDArray expected = createAppended(input, nrofZeros, 0);
        vertex.setInputs(input);
        assertEquals("Incorrect output!", expected, vertex.doForward(true, LayerWorkspaceMgr.noWorkspacesImmutable()));
    }

    /**
     * Test that the correct number of zeroes are appended
     */
    @Test
    public void doForward5D() {
        final long nrofZeros = 2;
        final GraphVertex vertex = new ode.vertex.conf.ConcatZerosVertex(nrofZeros)
                .instantiate(null, "test", 0, null, true);

        final INDArray input = Nd4j.ones(2, 3, 4, 5, 6);
        final INDArray expected = createAppended(input, nrofZeros, 0);
        vertex.setInputs(input);
        assertEquals("Incorrect output!", expected, vertex.doForward(true, LayerWorkspaceMgr.noWorkspacesImmutable()));
    }

    /**
     * Test that zeroes are removed from backwards pass
     */
    @Test
    public void doBackward2D() {
        final long nrofZeros = 5;
        final GraphVertex vertex = new ode.vertex.conf.ConcatZerosVertex(nrofZeros)
                .instantiate(null, "test", 0, null, true);

        final INDArray expected = Nd4j.ones(2, 3);
        final INDArray epsilon = createAppended(expected, nrofZeros, 666);
        vertex.setEpsilon(epsilon);
        assertEquals("Incorrect output!", expected, vertex.doBackward(true, LayerWorkspaceMgr.noWorkspacesImmutable()).getSecond()[0]);

    }

    /**
     * Test that zeroes are removed from backwards pass
     */
    @Test
    public void doBackward3D() {
        final long nrofZeros = 3;
        final GraphVertex vertex = new ode.vertex.conf.ConcatZerosVertex(nrofZeros)
                .instantiate(null, "test", 0, null, true);

        final INDArray expected = Nd4j.ones(2, 3, 4);
        final INDArray epsilon = createAppended(expected, nrofZeros, 666);
        vertex.setEpsilon(epsilon);
        assertEquals("Incorrect output!", expected, vertex.doBackward(true, LayerWorkspaceMgr.noWorkspacesImmutable()).getSecond()[0]);
    }

    /**
     * Test that zeroes are removed from backwards pass
     */
    @Test
    public void doBackward4D() {
        final long nrofZeros = 7;
        final GraphVertex vertex = new ode.vertex.conf.ConcatZerosVertex(nrofZeros)
                .instantiate(null, "test", 0, null, true);

        final INDArray expected = Nd4j.ones(2, 3, 4, 5);
        final INDArray epsilon = createAppended(expected, nrofZeros, 666);
        vertex.setEpsilon(epsilon);
        assertEquals("Incorrect output!", expected, vertex.doBackward(true, LayerWorkspaceMgr.noWorkspacesImmutable()).getSecond()[0]);
    }

    /**
     * Test that zeroes are removed from backwards pass
     */
    @Test
    public void doBackward5D() {
        final long nrofZeros = 2;
        final GraphVertex vertex = new ode.vertex.conf.ConcatZerosVertex(nrofZeros)
                .instantiate(null, "test", 0, null, true);

        final INDArray expected = Nd4j.ones(2, 3, 4, 5, 6);
        final INDArray epsilon = createAppended(expected, nrofZeros, 666);
        vertex.setEpsilon(epsilon);
        assertEquals("Incorrect output!", expected, vertex.doBackward(true, LayerWorkspaceMgr.noWorkspacesImmutable()).getSecond()[0]);
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
}