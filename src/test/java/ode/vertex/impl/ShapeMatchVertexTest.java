package ode.vertex.impl;


import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.impl.ElementWiseVertex;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collections;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link ShapeMatchVertex}
 *
 * @author Christian Skarby
 */
public class ShapeMatchVertexTest {

    /**
     * Test toString method
     */
    @Test
    public void toStringTest() {
        final int vertexIndex = 666;
        final String vertexName = "test";
        final GraphVertex vertex = new ElementWiseVertex(null, vertexName + "-vertex", vertexIndex, ElementWiseVertex.Op.Add);
        final GraphVertex test = new ode.vertex.conf.ShapeMatchVertex(
                new org.deeplearning4j.nn.conf.graph.ElementWiseVertex(
                        org.deeplearning4j.nn.conf.graph.ElementWiseVertex.Op.Add)).instantiate(
                                null, vertexName, vertexIndex, null, false);

        final String expected = ShapeMatchVertex.class.getSimpleName()
                + "(id=" + vertexIndex + ",name=\"" + vertexName + "\",vertex=" + vertex.toString() + ")";

        assertEquals("Incorrect name!", expected, test.toString());
    }

    /**
     * Test do forward with an {@link ElementWiseVertex}
     */
    @Test
    public void doForward() {
        final double scalar = 666;
        final INDArray input = Nd4j.arange(3*4*5*6).reshape(3,4,5,6);

        final GraphVertex vertex = new ode.vertex.conf.ShapeMatchVertex(
                new org.deeplearning4j.nn.conf.graph.ElementWiseVertex(
                        org.deeplearning4j.nn.conf.graph.ElementWiseVertex.Op.Add)).instantiate(
                null, "dummy", 1, null, false);
        vertex.setInputs(input, Nd4j.scalar(scalar));

        assertEquals("Incorrect output!",
                input.add(scalar),
                vertex.doForward(false, LayerWorkspaceMgr.noWorkspacesImmutable()));
    }

    /**
     * Test doBackward with a {@link ElementWiseVertex}
     */
    @Test
    public void doBackward() {
        final INDArray eps = Nd4j.arange(2*3*4).reshape(2,3,4);
        final GraphVertex vertex = new ShapeMatchVertex(null, "test", 1,
                new ElementWiseVertex(null, "test-vertex", 1, ElementWiseVertex.Op.Average), Collections.emptySet());

        vertex.setEpsilon(eps);
        vertex.setInputs(eps, eps, Nd4j.scalar(1));
        vertex.doForward(true ,LayerWorkspaceMgr.noWorkspacesImmutable()); // Needed to set inputs to do backward
        Pair<Gradient, INDArray[]> output = vertex.doBackward(true, LayerWorkspaceMgr.noWorkspacesImmutable());

        INDArray[] actualEps = output.getSecond();
        assertEquals("Incorrect epsilon!", eps.div(3), actualEps[0]);
        assertEquals("Incorrect epsilon!", eps.div(3), actualEps[1]);
        assertEquals("Incorrect epsilon!", eps.div(3).mean(), actualEps[2]);
    }
}