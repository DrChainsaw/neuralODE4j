package examples.spiral.vertex.impl;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

/**
 * Takes samples from a Gaussian process where mean and std are inputs, typically from a set of layers acting as a
 * variational auto encoder.
 *
 * @author Christian Skarby
 */
public class SampleGaussianVertex extends BaseGraphVertex {

    private final EpsSupplier rng;
    private INDArray lastEps;

    public interface EpsSupplier {
        INDArray get(long[] shape);
    }

    public SampleGaussianVertex(ComputationGraph graph, String name, int vertexIndex, EpsSupplier rng) {
        super(graph, name, vertexIndex, null, null);
        this.rng = rng;
    }

    @Override
    public INDArray doForward(boolean training, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoForward())
            throw new IllegalStateException("Cannot do forward pass: inputs not set");

        final INDArray mean = getInputs()[0];
        final INDArray logVar = getInputs()[1];

        lastEps = workspaceMgr.leverageTo(ArrayType.INPUT, rng.get(mean.shape()));

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, lastEps.mul(Transforms.exp(logVar.mul(0.5))).addi(mean));
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: errors not set");

        final INDArray epsMean = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, getEpsilon());

        // dL/dz * dz/dlogVar = epsilon * d/dlogVar(lastEps * e^0.5logVar + mean) = epsilon*0.5*lastEps*e^0.5logVar
        final INDArray logVar = getInputs()[1];
        final INDArray epsLogVar = getEpsilon().mul(lastEps).mul(0.5).muli(Transforms.exp(logVar.mul(0.5)));

        return new Pair<>(null, new INDArray[]{
                epsMean,
                workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsLogVar)
        });
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
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new IllegalArgumentException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        throw new UnsupportedOperationException("Not implemented yet!");
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + "(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\")";
    }
}
