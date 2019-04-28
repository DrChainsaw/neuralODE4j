package ode.vertex.impl;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

/**
 * Convenience class which concatenates zeros to inputs. Intended use case is to be augment an ODE by the process
 * described in https://arxiv.org/pdf/1904.01681.pdf.
 * <br><br>
 * The same thing can be achieved by many other means, e.g. adding an extra input to the graph.
 *
 * @author Christian Skarby
 */
public class ConcatZerosVertex extends BaseGraphVertex {

    private final long nrofZeros;


    public ConcatZerosVertex(ComputationGraph graph, String name, int vertexIndex, long nrofZeros) {
        super(graph, name, vertexIndex, null, null);
        this.nrofZeros = nrofZeros;
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
        final INDArray input = getInputs()[0];
        final long[] zeroesShape = input.shape().clone();

        zeroesShape[1] = nrofZeros; //Same dimension for all of CNNs, FF, RNNs

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, Nd4j.hstack(input, Nd4j.zeros(zeroesShape)));
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        final INDArray eps = getEpsilon();
        final INDArrayIndex[] epsView = NDArrayIndex.allFor(eps);
        epsView[1] = NDArrayIndex.interval(0, eps.size(1) - nrofZeros);

        return new Pair<>(null, new INDArray[]{workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, eps.get(epsView))});
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        // No parameters
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        throw new UnsupportedOperationException("Not supported!");
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + "(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\",nrofZeros=" + nrofZeros + ")";
    }
}
