package examples.anode;

import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.conditions.GreaterThanOrEqual;
import org.nd4j.linalg.indexing.conditions.LessThan;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

/**
 * Huber loss, aka smooth L1 loss. Uses an L2 loss for elements if the absolute element-wise error falls below 1 and an L1
 * loss otherwise.
 *
 * @author Christian Skarby
 */
public class LossHuber implements ILossFunction {

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
        final INDArray absDiff = Transforms.abs(labels.sub(output));

        final INDArray indsL2 = absDiff.cond(new LessThan(1));
        final INDArray sqr = indsL2.muli(absDiff).muli(absDiff).muli(0.5);

        final INDArray indsL1 = absDiff.cond(new GreaterThanOrEqual(1));

        final INDArray scoreArr = sqr.addi(absDiff.subi(0.5).muli(indsL1));

        if(mask != null) {
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
        final INDArray diff = output.subi(labels);
        final INDArray absDiff = Transforms.abs(diff);

        final INDArray indsL2 = absDiff.cond(new LessThan(1));
        final INDArray sqr = indsL2.muli(diff); // 0.5 and 2 cancel out

        final INDArray indsL1 = absDiff.cond(new GreaterThanOrEqual(1));

        final INDArray dLda = Transforms.sign(diff).muli(indsL1).addi(sqr);

        INDArray grad = activationFn.backprop(preOutput, dLda).getFirst();

        if(mask != null) {
            LossUtil.applyMask(grad, mask);
        }
        return grad;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask)
        );
    }

    @Override
    public String name() {
        return "LossHuber()";
    }
}
