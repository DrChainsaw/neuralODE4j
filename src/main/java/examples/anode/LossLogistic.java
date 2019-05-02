package examples.anode;

import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

/**
 * Logistic loss function. L = 1/ln(2) * ln(1 + e^(-labels * output)) where ln is natural logarithm
 *
 * @author Christian Skarby
 */
public class LossLogistic implements ILossFunction {

    private static final double ln2 = Math.log(2);
    private static final double ln2Inv = 1d / ln2;

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        final INDArray scoreArr = computeScoreArray(labels, preOutput, activationFn, mask);
        if(average) {
            return scoreArr.meanNumber().doubleValue();
        }
        return scoreArr.sumNumber().doubleValue();
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if(!labels.equalShapes(preOutput)){
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), preOutput.shape());
        }
        final INDArray output = activationFn.getActivation(preOutput.dup(), true);
        final INDArray scoreArr = Transforms.log(Transforms.exp(labels.neg().muli(output)).addi(1)).muli(ln2Inv);

        //Loss function with masking
        if (mask != null) {
            LossUtil.applyMask(scoreArr, mask);
        }

        return scoreArr;
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if(!labels.equalShapes(preOutput)){
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), preOutput.shape());
        }
        final INDArray output = activationFn.getActivation(preOutput.dup(), true);

        final INDArray dLda = labels.neg().divi(Transforms.exp(output.muli(labels)).addi(1).muli(ln2));

        if (mask != null && LossUtil.isPerOutputMasking(dLda, mask)) {
            //For *most* activation functions: we don't actually need to mask dL/da in addition to masking dL/dz later
            //but: some, like softmax, require both (due to dL/dz_i being a function of dL/da_j, for i != j)
            //We could add a special case for softmax (activationFn instanceof ActivationSoftmax) but that would be
            // error prone - but buy us a tiny bit of performance
            LossUtil.applyMask(dLda, mask);
        }

        INDArray gradients = activationFn.backprop(preOutput, dLda).getFirst();

        //Loss function with masking
        if (mask != null) {
            LossUtil.applyMask(gradients, mask);
        }

        return gradients;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        final double score = computeScore(labels, preOutput, activationFn, mask, average);
        final INDArray gradient = computeGradient(labels, preOutput, activationFn, mask);
        return new Pair<>(score, gradient);
    }

    @Override
    public String name() {
        return "LossLogistic()";
    }
}
