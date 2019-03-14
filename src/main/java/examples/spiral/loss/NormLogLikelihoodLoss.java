package examples.spiral.loss;

import lombok.Data;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Log-likelihood(ish?) loss under gaussian assumptions.
 * Reimplementation of https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py
 *
 * @author Christian Skarby
 */
@Data
public class NormLogLikelihoodLoss implements ILossFunction {

    private final static double log2pi = Math.log(2 * Math.PI);

    private final double logNoiseVar;

    public NormLogLikelihoodLoss(@JsonProperty("noiseSigma") double noiseSigma) {
        this.logNoiseVar = 2 * Math.log(noiseSigma);
    }

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

        if (!labels.equalShapes(output)) {
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), output.shape());
        }

        final INDArray predGrad = logNormalPdfGradient(labels, output);

        return activationFn.backprop(output, predGrad).getFirst();
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
    }

    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {

        if(mask != null) {
            // Should be straight forward to implement, but not needed
            throw new UnsupportedOperationException("Masking not supported");
        }

        final INDArray output = activationFn.getActivation(preOutput.dup(), true);

        final int[] sumDims = new int[output.rank()-1];
        for(int i = 0; i < sumDims.length; i++) {
            sumDims[i] = i+1;
        }

        if (!labels.equalShapes(output)) {
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), output.shape());
        }

        return logNormalPdf(labels, output).sum(sumDims).negi();
    }


    @Override
    public String name() {
        return "NormLogLikelihoodLoss(" + logNoiseVar +")";
    }

    private INDArray logNormalPdf(INDArray labels, INDArray output) {
        // Expression from original repo.
        // Similar to log-likelihood assuming parameters are from a gaussian distribution, but not 100% same?
        return Transforms.pow(output.rsub(labels), 2, false)
                .divi(Math.exp(logNoiseVar))
                .addi(log2pi)
                .addi(logNoiseVar)
                .muli(-0.5);
    }

    private INDArray logNormalPdfGradient(INDArray labels, INDArray output) {
        // 2 from derivative of exponent and -0.5 constant cancel out
        // Original implementation uses mean of loss along batch dimension
        return output.sub(labels).muli(Math.exp(-logNoiseVar)).divi(output.size(0));
    }
}
