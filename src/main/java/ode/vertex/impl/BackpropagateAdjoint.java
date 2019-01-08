package ode.vertex.impl;

import lombok.AllArgsConstructor;
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
import util.time.StatisticsTimer;

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

    private final AugmentedDynamics augmentedDynamics;
    private final FirstOrderEquation forwardPass;
    private final GraphInfo graphInfo;

    @AllArgsConstructor
    public static class GraphInfo {
        private final ComputationGraph graph;
        private final NonContiguous1DView realGradients;
        private final LayerWorkspaceMgr workspaceMgr;
        private final boolean truncatedBPTT;
    }

    final StatisticsTimer gradTimer = new StatisticsTimer();
    final StatisticsTimer updatePostTimer= new StatisticsTimer();
    final StatisticsTimer updatePreTimer= new StatisticsTimer();
    public BackpropagateAdjoint(
            AugmentedDynamics augmentedDynamics,
            FirstOrderEquation forwardPass,
            GraphInfo graphInfo) {
        this.augmentedDynamics = augmentedDynamics;
        this.forwardPass = forwardPass;
        this.graphInfo = graphInfo;
    }

    @Override
    public INDArray calculateDerivative(INDArray zAug, INDArray t, INDArray fzAug) {
        updatePreTimer.start();
        augmentedDynamics.updateFrom(zAug);
        updatePreTimer.stop();

        forwardPass.calculateDerivative(augmentedDynamics.z(), t, augmentedDynamics.z());

        try (WorkspacesCloseable ws = graphInfo.workspaceMgr.notifyScopeEntered(ArrayType.ACTIVATIONS, ArrayType.ACTIVATION_GRAD)) {

            // Seems like some layers let previous gradients influence their new gradients. I haven't really figured out
            // why but it seems to have a detrimental effect on the accuracy and general stability
            graphInfo.graph.getFlattenedGradients().assign(0);

            gradTimer.start();
            final List<INDArray> ret = backPropagate(augmentedDynamics.zAdjoint().negi());
            gradTimer.stop();

            updatePostTimer.start();
            augmentedDynamics.updateZAdjoint(ret);
            graphInfo.realGradients.assignTo(augmentedDynamics.paramAdjoint());
            augmentedDynamics.tAdjoint().assign(0); // Nothing depends on t as of yet.
        }

        augmentedDynamics.transferTo(fzAug);
        updatePostTimer.stop();
        return fzAug;
    }

    private List<INDArray> backPropagate(INDArray epsilon) {

        //Do backprop, in reverse topological order
        final int[] topologicalOrder = graphInfo.graph.topologicalSortOrder();
        final GraphVertex[] vertices = graphInfo.graph.getVertices();

        List<INDArray> outputEpsilons = new ArrayList<>();

        boolean[] setVertexEpsilon = new boolean[topologicalOrder.length]; //If true: already set epsilon for this vertex; later epsilons should be *added* to the existing one, not set
        for (int i = topologicalOrder.length - 1; i >= 0; i--) {
            GraphVertex current = vertices[topologicalOrder[i]];

            if (current.isOutputVertex()) {
                for (VertexIndices vertexIndices : current.getInputVertices()) {
                    final String inputName = vertices[vertexIndices.getVertexIndex()].getVertexName();
                    graphInfo.graph.getVertex(inputName).setEpsilon(epsilon);
                }
                continue;
            }

            if (current.isInputVertex()) {
                continue;
            }

            Pair<Gradient, INDArray[]> pair;
            INDArray[] epsilons;
            pair = current.doBackward(graphInfo.truncatedBPTT, graphInfo.workspaceMgr);
            epsilons = pair.getSecond();

            for (VertexIndices vertexIndices : current.getInputVertices()) {
                final String inputName = vertices[vertexIndices.getVertexIndex()].getVertexName();
                if (graphInfo.graph.getConfiguration().getNetworkInputs().contains(
                        inputName)) {
                    outputEpsilons.add(graphInfo.graph.getConfiguration().getNetworkInputs().indexOf(inputName),
                            epsilons[vertexIndices.getVertexEdgeNumber()]);
                }
            }

            //Inputs to the current GraphVertex:
            VertexIndices[] inputVertices = current.getInputVertices();

            //Set epsilons for the vertices that provide inputs to this vertex:
            if (inputVertices != null) {
                int j = 0;
                for (VertexIndices v : inputVertices) {
                    GraphVertex gv = graphInfo.graph.getVertices()[v.getVertexIndex()];
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

        return outputEpsilons;
    }
}
