package examples.spiral.vertex.impl;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
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

        final INDArray input = getInputs()[0].dup();
        final long size = input.size(1) / 2;

        // Dup due to dl4j issue #7263
        INDArray mean = input.get(NDArrayIndex.all(), NDArrayIndex.interval(0, size)).dup();
        INDArray logVar = input.get(NDArrayIndex.all(), NDArrayIndex.interval(size, 2 * size)).dup();

        lastEps = rng.get(mean.shape()).mul(Transforms.exp(logVar.mul(0.5)));
        if (training) {
            lastEps = workspaceMgr.leverageTo(ArrayType.INPUT, lastEps);
        }

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, lastEps.add(mean));
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: errors not set");

        final INDArray epsMean = getEpsilon();

        // dL/dz * dz/dlogVar = epsilon * d/dlogVar(lastEps * e^0.5logVar + mean) = epsilon*0.5*lastEps*e^0.5logVar
        final INDArray epsLogVar = getEpsilon().dup().mul(lastEps).mul(0.5);

        final INDArray combinedEps = Nd4j.hstack(epsMean, epsLogVar);
        return new Pair<>(null, new INDArray[]{
                workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, combinedEps)
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
