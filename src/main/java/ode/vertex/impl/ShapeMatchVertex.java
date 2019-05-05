package ode.vertex.impl;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import util.preproc.DuplicateScalarToShape;

import java.util.Map;

/**
 * Duplicates the last input to match the shapes of the other inputs. Main use case is for performing merging or element
 * wise operations with current time from an ODE solver.
 *
 * @author Christian Skarby
 */
public class ShapeMatchVertex extends BaseGraphVertex {

    private final GraphVertex vertex;
    protected Map<Integer, Long> overrideSizeDims;

    public ShapeMatchVertex(
            ComputationGraph graph, String name, int vertexIndex, DataType networkDataType,
            GraphVertex vertex, Map<Integer, Long> overrideSizeDims) {
        super(graph, name, vertexIndex, null, null, networkDataType);
        this.vertex = vertex;
        this.overrideSizeDims = overrideSizeDims;
    }


    @Override
    public String toString() {
        return this.getClass().getSimpleName() + "(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\",vertex=" + vertex.toString() + ")";
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
        if (!canDoForward())
            throw new IllegalStateException("Cannot do forward pass: inputs not set");

        final INDArray[] inputs = getInputs().clone();

        final long[] shape = inputs[0].shape().clone();

        for(Map.Entry<Integer, Long> sizeDim: overrideSizeDims.entrySet()) {
            shape[sizeDim.getKey()] = sizeDim.getValue();
        }

        final INDArray newLast = new DuplicateScalarToShape(shape)
                .preProcess(inputs[inputs.length - 1], -1, workspaceMgr);

        inputs[inputs.length - 1] = newLast;

        vertex.setInputs(inputs);
        return vertex.doForward(training, workspaceMgr);
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        vertex.setEpsilon(getEpsilon());
        final Pair<Gradient, INDArray[]> output = vertex.doBackward(tbptt, workspaceMgr);

        final INDArray[] epsilons = output.getSecond();
        epsilons[epsilons.length - 1] = new DuplicateScalarToShape().backprop(epsilons[epsilons.length - 1], -1, workspaceMgr);
        return output;
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        // No gradients
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        return null;
    }
}
