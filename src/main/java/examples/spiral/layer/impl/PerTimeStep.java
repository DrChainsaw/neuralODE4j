package examples.spiral.layer.impl;


import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;

/**
 * Comprises a sequence of {@link Layer}s which will be activated once per time step in the input. Basically a poor mans
 * version of pytorchs Linear which accepts inputs of any dimension as long as the size of the last dimension is the
 * same as the number of inputs to the layer. This layer only accepts 3D inputs and returns outputs of one higher
 * dimension than the last layer in the sequence.
 *
 * @author Christian Skarby
 */
public class PerTimeStep extends BaseLayer<examples.spiral.layer.conf.PerTimeStep> {

    private final List<Layer> layers;

    private static final int BATCH_DIM = 0;
    private static final int SIZE_DIM = 1;
    private static final int TIME_DIM = 2;

    public PerTimeStep(NeuralNetConfiguration conf, List<Layer> layers) {
        super(conf);
        this.layers = layers;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        return null;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        final INDArray input = getInput();
        INDArray output = null;
        final long nrofSteps = input.size(TIME_DIM);

        for(int i = 0; i < nrofSteps; i++) {
            INDArray activation = input.tensorAlongDimension(i, BATCH_DIM, SIZE_DIM);
            for(Layer layer: layers) {
                activation = layer.activate(activation, true, workspaceMgr);
            }

            if(output == null) {
                output = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, activation.size(BATCH_DIM), activation.size(SIZE_DIM), nrofSteps);
            }
            output.tensorAlongDimension(i, BATCH_DIM, SIZE_DIM).assign(activation);
        }
        return output;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public void clearNoiseWeightParams() {
        for(Layer layer: layers) {
            layer.clearNoiseWeightParams();
        }
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        super.setBackpropGradientsViewArray(gradients);
        long paramCnt = 0;
        for(Layer layer: layers) {
            final long numPars = layer.numParams(true);
            final INDArray gradView = gradients.get(NDArrayIndex.point(0), NDArrayIndex.interval(paramCnt, paramCnt + numPars));
            layer.setBackpropGradientsViewArray(gradView);
            paramCnt += numPars;
        }
    }
}
