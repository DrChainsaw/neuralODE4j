package examples.spiral.vertex.impl;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * Makes 3D input 2D by making dim 2 (typically time) part of dim 0 (batch dim). Basically a poor mans version of
 * torch.nn.Linear wrt allowing 3D input to Dense layers.
 *
 * @author Christian Skarby
 */
public class TimeAsBatch extends BaseGraphVertex {

    private static final int BATCH_DIM = 0;
    private static final int SIZE_DIM = 1;
    private static final int TIME_DIM = 2;

    private long nrofTimeSteps;

    public TimeAsBatch(ComputationGraph graph, String name, int vertexIndex) {
        this(graph, name, vertexIndex, null, null);
    }

    protected TimeAsBatch(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices, VertexIndices[] outputVertices) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
    }

    @Override
    public String toString() {
        return "TimeAsBatch()";
    }

    @Override
    public boolean hasLayer() {
        return false;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training, LayerWorkspaceMgr workspaceMgr) {
        if(getInputs().length != 1) {
            throw new IllegalStateException("Only one input array supported!");
        }

        final INDArray input = getInputs()[0];
        nrofTimeSteps = input.size(TIME_DIM);
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS,
                input.permute(BATCH_DIM,TIME_DIM,SIZE_DIM).reshape(input.size(BATCH_DIM)*input.size(TIME_DIM), input.size(SIZE_DIM)));
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {

        final INDArray input = getEpsilon();
        return new Pair<>(null, new INDArray[]{workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD,
                input.reshape(input.size(BATCH_DIM) / nrofTimeSteps, nrofTimeSteps, input.size(SIZE_DIM)).permutei(BATCH_DIM, TIME_DIM, SIZE_DIM))});
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {

    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        throw new UnsupportedOperationException("Not implemented!");
    }
}
