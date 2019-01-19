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
     * Return the solution to the ODE for the given inputs (typically activations from previous layers)
     * @param wsMgr To handle workspaces for newly created arrays
     * @param inputs Inputs to vertex
     * @return an {@link INDArray} with the solution to the ODE which is typically the output
     */
    INDArray solve(ComputationGraph graph, LayerWorkspaceMgr wsMgr, INDArray[] inputs);
}
