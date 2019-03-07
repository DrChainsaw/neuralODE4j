package examples.spiral.loss;

import lombok.Data;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

/**
 * Kullback-Leibler divergence assuming gaussian distribution
 * Reimplementation of https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py
 *
 * @author Christian Skarby
 */
@Data
public class NormKLDLoss implements ILossFunction {

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        if(average) {
            return scoreArray(labels, preOutput, activationFn, mask).meanNumber().doubleValue();
        }
        return scoreArray(labels, preOutput, activationFn, mask).sumNumber().doubleValue();
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        return scoreArray(labels, preOutput, activationFn, mask);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        final INDArray output = activationFn.getActivation(preOutput.dup(), true);

        long size = output.size(1) / 2;
        INDArray mean = output.get(NDArrayIndex.all(), NDArrayIndex.interval(0, size)).dup();
        INDArray logVar = output.get(NDArrayIndex.all(), NDArrayIndex.interval(size, 2 * size)).dup();

        final INDArray meanGrad = normalKlGradMu1(mean);
        final INDArray logvarGrad = normalKlGradLv1(logVar);

        return activationFn.backprop(output, Nd4j.hstack(meanGrad, logvarGrad)).getFirst();
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
    }

    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {

        final INDArray output = activationFn.getActivation(preOutput.dup(), true);

        long size = output.size(1) / 2;
        INDArray mean = output.get(NDArrayIndex.all(), NDArrayIndex.interval(0, size)).dup();
        INDArray logVar = output.get(NDArrayIndex.all(), NDArrayIndex.interval(size, 2 * size)).dup();

        return normalKl(mean, logVar).sum(1);
    }

    @Override
    public String name() {
        return "NormKLDLoss()";
    }

    private INDArray normalKl(INDArray mu1, INDArray lv1 ) {
        final double mu2 = 0;
        final INDArray v1 = Transforms.exp(lv1);
        final double v2 = 1;
        final INDArray lstd1 = lv1.div(2);
        final double lstd2 = 0;
        return lstd1.rsubi(lstd2).addi(
                v1.addi(Transforms.pow(mu1.sub(mu2), 2, false)).divi(2 * v2)
        ).subi(0.5);
    }

    private INDArray normalKlGradMu1(INDArray mu1) {
        final double mu2 = 0;
        final double v2 = 1;
        return mu1.sub(mu2).muli(v2);
    }

    private INDArray normalKlGradLv1(INDArray lv1) {
        final double lv2 = 0;
        return Transforms.exp(lv1.sub(lv2)).subi(1).divi(2);
    }
}
