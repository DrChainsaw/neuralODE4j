package examples.spiral.layer.conf;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.RnnLossLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link PerTimeStep}
 *
 * @author Christian Skarby
 */
public class PerTimeStepTest {

    /**
     * Test activation of two layers. First layer multiplies input by 2 and second layer multiplies input by 3. Expected
     * outcome is thus that input has been multiplied by a factor of 6.
     */
    @Test
    public void activate() {
        final long nrofInputs = 4;
        final long nrofTimeSteps = 10;
        final MultiLayerNetwork network = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.IDENTITY)
                .list()
                .setInputType(InputType.recurrent(nrofInputs, nrofTimeSteps))
                .layer(new PerTimeStep.Builder()
                        .addLayer(new DenseLayer.Builder().nOut(nrofInputs).hasBias(false).activation(new ActivationIdentity()).build())
                        .addLayer(new DenseLayer.Builder().nOut(nrofInputs).hasBias(false).activation(new ActivationIdentity()).build())
                        .build())
                .layer(new RnnLossLayer.Builder().activation(new ActivationIdentity()).hasBias(false).build())
                .build());
        network.init();
        final long size = network.getLayer(0).params().length();
        network.getLayer(0).params().get(NDArrayIndex.point(0), NDArrayIndex.interval(0, size/2)).muli(2);
        network.getLayer(0).params().get(NDArrayIndex.point(0), NDArrayIndex.interval(size/2, size)).muli(3);

        final long batchSize = 3;
        final INDArray input = Nd4j.arange(batchSize*nrofInputs*nrofTimeSteps).reshape(batchSize, nrofInputs, nrofTimeSteps);
        final INDArray expected = input.mul(6);

        final INDArray actual = network.output(input);

        assertEquals("Incorrect output!", expected, actual);
    }

}