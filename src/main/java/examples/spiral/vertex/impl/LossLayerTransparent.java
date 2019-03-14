package examples.spiral.vertex.impl;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;

/**
 * Loss layer which is transparent in the sense that it just forwards its input to a given {@link ILossFunction}.
 *
 * Copy paste of {@link org.deeplearning4j.nn.layers.recurrent.RnnLossLayer} but with reshaping to 2D removed.
 *
 * @author Christian Skarby
 */
public class LossLayerTransparent extends BaseLayer<examples.spiral.vertex.conf.LossLayerTransparent> implements IOutputLayer {
    @Setter @Getter protected INDArray labels;

    public LossLayerTransparent(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if (input.rank() != 3)
            throw new UnsupportedOperationException(
                    "Input is not rank 3. Got input with rank " + input.rank() + " " + layerId());
        if (labels == null)
            throw new IllegalStateException("Labels are not set (null)");

        final INDArray inputLev = workspaceMgr.dup(ArrayType.BP_WORKING_MEM, input);
        final INDArray labelsLev = workspaceMgr.dup(ArrayType.BP_WORKING_MEM, labels);

        // delta calculation
        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray delta = lossFunction.computeGradient(labelsLev, inputLev, layerConf().getActivationFn(), maskArray);

        // grab the empty gradient
        Gradient gradient = new DefaultGradient();
        return new Pair<>(gradient, workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, delta));
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double f1Score(DataSet data) {
        return 0;
    }

    /**{@inheritDoc}
     */
    @Override
    public double f1Score(INDArray examples, INDArray labels) {
        INDArray out = activate(examples, false, null);
        Evaluation eval = new Evaluation();
        eval.evalTimeSeries(labels, out, maskArray);
        return eval.f1();
    }

    @Override
    public int numLabels() {
        // FIXME: int cast
        return (int) labels.size(1);
    }

    @Override
    public void fit(DataSetIterator iter) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public int[] predict(INDArray examples) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public List<String> predict(DataSet dataSet) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public INDArray labelProbabilities(INDArray examples) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void fit(INDArray examples, INDArray labels) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void fit(DataSet data) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void fit(INDArray examples, int[] labels) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public Type type() {
        return Type.RECURRENT;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        if (input.rank() != 3)
            throw new UnsupportedOperationException(
                    "Input must be rank 3. Got input with rank " + input.rank() + " " + layerId());

        INDArray out = layerConf().getActivationFn().getActivation(workspaceMgr.dup(ArrayType.ACTIVATIONS, input, input.ordering()), training);
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, out);
    }

    @Override
    public void setMaskArray(INDArray maskArray) {
        this.maskArray = maskArray;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                                                          int minibatchSize) {
        this.maskArray = maskArray;   //TODO
        this.maskState = currentMaskState;

        return null; //Last layer in network
    }

    @Override
    public boolean needsLabels() {
        return true;
    }

    @Override
    public double computeScore(double fullNetworkL1, double fullNetworkL2, boolean training, LayerWorkspaceMgr workspaceMgr) {
        ILossFunction lossFunction = layerConf().getLossFn();

        double score = lossFunction.computeScore(labels, input.dup(), layerConf().getActivationFn(), maskArray,false);
        score += fullNetworkL1 + fullNetworkL2;
        score /= getInputMiniBatchSize();

        this.score = score;

        return score;
    }

    /**Compute the score for each example individually, after labels and input have been set.
     *
     * @param fullNetworkL1 L1 regularization term for the entire network (or, 0.0 to not include regularization)
     * @param fullNetworkL2 L2 regularization term for the entire network (or, 0.0 to not include regularization)
     * @return A column INDArray of shape [numExamples,1], where entry i is the score of the ith example
     */
    @Override
    public INDArray computeScoreForExamples(double fullNetworkL1, double fullNetworkL2, LayerWorkspaceMgr workspaceMgr) {
        //For RNN: need to sum up the score over each time step before returning.

        if (input == null || labels == null)
            throw new IllegalStateException("Cannot calculate score without input and labels " + layerId());

        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray scoreArray =
                lossFunction.computeScoreArray(labels, input, layerConf().getActivationFn(), maskArray);

        INDArray summedScores = scoreArray.sum(1,2);

        double l1l2 = fullNetworkL1 + fullNetworkL2;
        if (l1l2 != 0.0) {
            summedScores.addi(l1l2);
        }

        return summedScores;
    }
}
