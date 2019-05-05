package examples.spiral.loss;

import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link NormLogLikelihoodLoss}
 *
 * @author Christian Skarby
 */
public class NormLogLikelihoodLossTest {

    private final double[][][] expectedGrad = {{{-29.2222, -27.2069, -25.1916, -23.1762, -21.1609},
            {-19.1456, -17.1303, -15.1149, -13.0996, -11.0843}},

            {{-9.0690, -7.0536, -5.0383, -3.0230, -1.0077},
                    {1.0077, 3.0230, 5.0383, 7.0536, 9.0690}},

            {{11.0843, 13.0996, 15.1149, 17.1303, 19.1456},
                    {21.1609, 23.1762, 25.1916, 27.2069, 29.2222}}};

    /**
     * Test that score and gradient is zero when label and prediction are the same
     */
    @Test
    public void computeGradientAndScorePerfectMatch() {
        final INDArray traj = Nd4j.linspace(-3.45, 2.34, 2 * 3 * 5, DataType.FLOAT).reshape(3, 2, 5);

        final Pair<Double, INDArray> out = new NormLogLikelihoodLoss(0.3)
                .computeGradientAndScore(traj, traj, new ActivationIdentity(), null, false);

        // Loss has a constant term
        assertEquals("Expected minimum loss!", -8.55102825164795, out.getFirst(), 1e-5);
        assertEquals("Expected zero grad!", 0, out.getSecond().amaxNumber().doubleValue(), 1e-10);
    }

    /**
     * Test that score and gradient is correct when label and prediction are not the same. Numbers taken from original repo.
     */
    @Test
    public void computeGradientAndScoreWithEps() {
        final INDArray traj = Nd4j.linspace(-3.45, 2.34, 2 * 3 * 5, DataType.FLOAT).reshape(3, 2, 5);
        final INDArray eps = Nd4j.linspace(-7.89, 7.89, traj.length(), DataType.FLOAT).reshape(traj.shape());

        final Pair<Double, INDArray> out = new NormLogLikelihoodLoss(0.3)
                .computeGradientAndScore(traj, traj.add(eps), new ActivationIdentity(), null, true);


        out.getSecond().divi(3); // Dl4j scales gradients with mini batch size centrally in BaseMultiLayerUpdater
        assertEquals("Incorrect loss!", 1229.4709, out.getFirst(), 1e-4);

        for (int i = 0; i < expectedGrad.length; i++) {
            for (int j = 0; j < expectedGrad[i].length; j++) {
                assertArrayEquals("Incorrect gradient along " + i + ", " + j + "!",
                        expectedGrad[i][j],
                        out.getSecond().get(NDArrayIndex.point(i), NDArrayIndex.point(j), NDArrayIndex.all()).toDoubleVector(), 1e-4);
            }
        }
    }
}