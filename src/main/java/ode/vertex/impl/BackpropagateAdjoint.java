package ode.vertex.impl;

import ode.solve.api.FirstOrderEquation;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.workspace.WorkspacesCloseable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Models back propagation through a undefined number of residual blocks as a first order differential equation using
 * adjoint state.
 * See https://arxiv.org/pdf/1806.07366.pdf
 * <br><br>
 * Interpretation of augmented dynamics for DL4J:
 * <br><br>
 * <pre>
 * f(z(t), theta) = output from forward pass through the layers of the ODE vertex (i.e. the layers of graph)
 * -a(t)*df/dz(t) = dL / dz(t) = epsilon from a backward pass through the layers of the ODE vertex (i.e. the layers of graph) wrt previous output (is it really?)
 * -a(t) * df / dt = no change (not used yet, maybe set to 0?)
 * -a(t) df/dtheta = -dL / dtheta = Gradient from a backward pass through the layers of the ODE vertex (i.e. the layers of graph) wrt -epsilon
 *</pre>
 * @author Christian Skarby
 */
public class BackpropagateAdjoint implements FirstOrderEquation {

    private static final Logger log = LoggerFactory.getLogger(BackpropagateAdjoint.class);

    private final ComputationGraph graph;
    private final LayerWorkspaceMgr workspaceMgr;
    private final AugmentedDynamics augmentedDynamics;
    private final FirstOrderEquation forwardPass;
    private final boolean truncatedBPTT;

    public BackpropagateAdjoint(
            ComputationGraph graph,
            LayerWorkspaceMgr workspaceMgr,
            AugmentedDynamics augmentedDynamics,
            FirstOrderEquation forwardPass,
            boolean truncatedBPTT) {
        this.graph = graph;
        this.workspaceMgr = workspaceMgr;
        this.augmentedDynamics = augmentedDynamics;
        this.forwardPass = forwardPass;
        this.truncatedBPTT = truncatedBPTT;
    }

    @Override
    public INDArray calculateDerivative(INDArray zAug, INDArray t, INDArray fzAug) {
        augmentedDynamics.updateFrom(zAug);

        forwardPass.calculateDerivative(augmentedDynamics.getZ(), t, augmentedDynamics.getZ());

        try (WorkspacesCloseable ws = workspaceMgr.notifyScopeEntered(ArrayType.ACTIVATIONS, ArrayType.ACTIVATION_GRAD)) {

            final INDArray prevFlattenedGrads = graph.getFlattenedGradients().dup();

            final List<INDArray> ret = backPropagate(augmentedDynamics.getZAdjoint().negi());

            augmentedDynamics.updateZAdjoint(ret);
            augmentedDynamics.updateParamAdjoint(graph.getFlattenedGradients());

            graph.getFlattenedGradients().assign(prevFlattenedGrads);
        }

        augmentedDynamics.transferTo(fzAug);
        return fzAug;
    }

    private List<INDArray> backPropagate(INDArray epsilon) {

        //Do backprop, in reverse topological order
        final int[] topologicalOrder = graph.topologicalSortOrder();
        final GraphVertex[] vertices = graph.getVertices();

        List<INDArray> outputEpsilons = new ArrayList<>();

        boolean[] setVertexEpsilon = new boolean[topologicalOrder.length]; //If true: already set epsilon for this vertex; later epsilons should be *added* to the existing one, not set
        for (int i = topologicalOrder.length - 1; i >= 0; i--) {
            GraphVertex current = vertices[topologicalOrder[i]];

            if (current.isOutputVertex()) {
                for (VertexIndices vertexIndices : current.getInputVertices()) {
                    final String inputName = vertices[vertexIndices.getVertexIndex()].getVertexName();
                    graph.getVertex(inputName).setEpsilon(epsilon);
                }
                continue;
            }

            if (current.isInputVertex()) {
                continue;
            }

            Pair<Gradient, INDArray[]> pair;
            INDArray[] epsilons;
            pair = current.doBackward(truncatedBPTT, workspaceMgr);
            epsilons = pair.getSecond();

            if (log.isWarnEnabled()) {
                final double max = pair.getFirst().gradient().maxNumber().doubleValue();
                if (max > 50) {
                    log.warn(current.getVertexName() + " large gradient found: " + max);
                }
            }

            for (VertexIndices vertexIndices : current.getInputVertices()) {
                final String inputName = vertices[vertexIndices.getVertexIndex()].getVertexName();
                if (graph.getConfiguration().getNetworkInputs().contains(
                        inputName)) {
                    outputEpsilons.add(graph.getConfiguration().getNetworkInputs().indexOf(inputName),
                            epsilons[vertexIndices.getVertexEdgeNumber()].migrate(true));
                }
            }

            //Inputs to the current GraphVertex:
            VertexIndices[] inputVertices = current.getInputVertices();

            //Set epsilons for the vertices that provide inputs to this vertex:
            if (inputVertices != null) {
                int j = 0;
                for (VertexIndices v : inputVertices) {
                    GraphVertex gv = graph.getVertices()[v.getVertexIndex()];
                    if (setVertexEpsilon[gv.getVertexIndex()]) {
                        //This vertex: must output to multiple vertices... we want to add the epsilons here
                        INDArray currentEps = gv.getEpsilon();
                        gv.setEpsilon(currentEps.addi(epsilons[j++]));  //TODO is this always safe?
                    } else {
                        gv.setEpsilon(epsilons[j++]);
                    }
                    setVertexEpsilon[gv.getVertexIndex()] = true;
                }
            }
        }

        //Now, add the gradients in the order we need them in for flattening (same as params order)

        return outputEpsilons;
    }
}
