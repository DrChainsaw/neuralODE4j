package ode.vertex.impl;

import ode.solve.api.FirstOrderEquation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.jetbrains.annotations.Nullable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.workspace.WorkspacesCloseable;

/**
 * Models forward pass through a undefined number of residual blocks as a first order differential equation.
 * See https://arxiv.org/pdf/1806.07366.pdf
 *
 * @author Christian Skarby
 */
public class ForwardPass implements FirstOrderEquation {

    private final ComputationGraph graph;
    private final LayerWorkspaceMgr innerWorkspaceMgr;
    private final boolean training;
    private final INDArray[] inputs;

    public ForwardPass(ComputationGraph graph,
                       LayerWorkspaceMgr innerWorkspaceMgr,
                       boolean training,
                       INDArray[] startInputs) {
        this.graph = graph;
        this.innerWorkspaceMgr = innerWorkspaceMgr;
        this.training = training;
        this.inputs = startInputs;
    }


    @Override
    public INDArray calculateDerivative(INDArray y, INDArray t, INDArray fy) {
            try (WorkspacesCloseable wsCloseable = innerWorkspaceMgr.notifyScopeEntered(ArrayType.ACTIVATIONS, ArrayType.INPUT)) {
                setInputsFromFlat(y);
                evaluate(inputs, fy);
            }

        return fy;
    }

    private void setInputsFromFlat(INDArray flatArray) {
        int lastInd = 0;
        for (int i = 0; i < inputs.length; i++) {
            INDArray input = inputs[i];
            final INDArray z = flatArray.get(NDArrayIndex.interval(lastInd, lastInd + input.length()));
            lastInd += input.length();
            inputs[i].assign(z.reshape(input.shape()));
        }
    }

    @Nullable
    private void evaluate(INDArray[] inputs, INDArray output) {
        //TODO: Might want to have internal workspace handling to conserve memory
        final int[] topologicalOrder = graph.topologicalSortOrder();
        final NDArrayIndexAccumulator outputAccum = new NDArrayIndexAccumulator(output);

        int outputCnt = 0;
        //Do forward pass according to the topological ordering of the network
        for (int i = 0; i <= graph.getVertices().length - 1; i++) {
            GraphVertex current = graph.getVertices()[topologicalOrder[i]];
            int vIdx = current.getVertexIndex();

            VertexIndices[] inputsTo = current.getOutputVertices();

            INDArray out = null;
            if (current.isInputVertex()) {
                out = inputs[vIdx];
            } else if (current.isOutputVertex()) {
                for (INDArray outArr : current.getInputs()) {
                    outputAccum.increment(outArr);
                    outputCnt++;
                }
            } else {
                //Standard feed-forward case
                out = current.doForward(training, innerWorkspaceMgr);
            }

            if (inputsTo != null) {  //Output vertices may not input to any other vertices
                for (VertexIndices v : inputsTo) {
                    //Note that we don't have to do anything special here: the activations are always detached in
                    // this method
                    int inputToIndex = v.getVertexIndex();
                    int vIdxEdge = v.getVertexEdgeNumber();
                    graph.getVertices()[inputToIndex].setInput(vIdxEdge, out.detach(), innerWorkspaceMgr);
                }
            }
        }
    }

}
