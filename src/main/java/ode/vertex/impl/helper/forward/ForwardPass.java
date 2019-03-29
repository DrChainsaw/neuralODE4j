package ode.vertex.impl.helper.forward;

import ode.solve.api.FirstOrderEquation;
import ode.vertex.impl.helper.NDArrayIndexAccumulator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.jetbrains.annotations.Nullable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.workspace.WorkspacesCloseable;

import java.util.ArrayList;
import java.util.List;

/**
 * Models forward pass through an undefined number of residual blocks as a first order differential equation.
 * See https://arxiv.org/pdf/1806.07366.pdf
 *
 * @author Christian Skarby
 */
public class ForwardPass implements FirstOrderEquation {

    private final ComputationGraph graph;
    private final LayerWorkspaceMgr workspaceMgr;
    private final boolean training;
    private final GraphInput input;

    public ForwardPass(ComputationGraph graph,
                       LayerWorkspaceMgr workspaceMgr,
                       boolean training,
                       GraphInput input) {
        this.graph = graph;
        this.workspaceMgr = workspaceMgr;
        this.training = training;
        this.input = input;
    }

    @Override
    public INDArray calculateDerivative(INDArray y, INDArray t, INDArray fy) {
        graph.getConfiguration().setIterationCount(graph.getIterationCount() + 1);
        try (WorkspacesCloseable ws = enterIfNotOpen(ArrayType.ACTIVATIONS)) {
            final INDArray[] inputs = input.getInputsFrom(y ,t);
            evaluate(inputs, fy);
        }
        return fy;
    }

    private WorkspacesCloseable enterIfNotOpen(ArrayType... types) {
        List<ArrayType> shallOpen = new ArrayList<>();
        for(ArrayType type: types) {
            if (!workspaceMgr.isWorkspaceOpen(type)) {
                shallOpen.add(type);
            }
        }

        return workspaceMgr.notifyScopeEntered(shallOpen.toArray(new ArrayType[0]));
    }

    @Nullable
    private void evaluate(INDArray[] inputs, INDArray output) {
        //TODO: Might want to have internal workspace handling to conserve memory
        final int[] topologicalOrder = graph.topologicalSortOrder();
        final NDArrayIndexAccumulator outputAccum = new NDArrayIndexAccumulator(output);

        //Do forward pass according to the topological ordering of the network
        for (int i = 0; i <= graph.getVertices().length - 1; i++) {
            GraphVertex current = graph.getVertices()[topologicalOrder[i]];
            int vIdx = current.getVertexIndex();

            VertexIndices[] inputsTo = current.getOutputVertices();

            final INDArray out;
            if (current.isInputVertex()) {
                out = inputs[vIdx];
            } else {
                //Standard feed-forward case
                out = current.doForward(training, workspaceMgr);
            }

            if (inputsTo == null) {  //Output vertices may not input to any other vertices
                outputAccum.increment(out);
            } else {
                for (VertexIndices v : inputsTo) {
                    //Note that we don't have to do anything special here: the activations are always detached in
                    // this method
                    int inputToIndex = v.getVertexIndex();
                    int vIdxEdge = v.getVertexEdgeNumber();
                    GraphVertex outputVertex = graph.getVertices()[inputToIndex];
                    if (outputVertex.getInputs() == null || outputVertex.getInputs()[vIdxEdge] == null) {
                        outputVertex.setInput(vIdxEdge, workspaceMgr.leverageTo(ArrayType.INPUT, out), workspaceMgr);
                    } else {
                        outputVertex.getInputs()[vIdxEdge].assign(out);
                    }

                }
            }
        }
    }

}
