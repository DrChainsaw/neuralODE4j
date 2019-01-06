package examples.mnist;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import static examples.mnist.LayerUtil.norm;

/**
 * Adds output block
 *
 * @author Christian Skarby
 */
public class Output implements Block {

    private final int nrofKernels;

    public Output(int nrofKernels) {
        this.nrofKernels = nrofKernels;
    }

    @Override
    public String add(String prev, ComputationGraphConfiguration.GraphBuilder builder) {
        builder
                .addLayer("normOutput",
                        norm(nrofKernels), prev)
                .addLayer("globPool", new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build(), "normOutput")
                .addLayer("output", new OutputLayer.Builder()
                        .nOut(10)
                        .lossFunction(new LossMCXENT())
                        .activation(new ActivationSoftmax()).build(), "globPool")
                .setOutputs("output");
        return "output";
    }
}
