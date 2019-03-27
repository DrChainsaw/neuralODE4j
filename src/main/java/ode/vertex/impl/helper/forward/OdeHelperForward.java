package ode.vertex.impl.helper.forward;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Helps with input/output handling when solving ODEs inside a neural network
 *
 * @author Christian Skarby
 */
public interface OdeHelperForward {

    /**
     * Return the solution to the ODE when assuming that a forward pass through the layers of the given graph is
     * the derivative of the sought function.
     * @param graph Graph of layers to do forward pass through
     * @param wsMgr To handle workspaces for newly created arrays
     * @param inputs Inputs to vertex, typically activations from previous layers
     * @return an {@link INDArray} with the solution to the ODE
     */
    INDArray solve(ComputationGraph graph, LayerWorkspaceMgr wsMgr, INDArray[] inputs);
}
