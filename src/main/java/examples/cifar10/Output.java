package examples.cifar10;

import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

/**
 * Adds output block for CIFAR 10
 *
 * @author Christian Skarby
 */
public class Output implements Block {

    @Override
    public String add(GraphBuilderWrapper builder, String ... prev) {
        builder
                .addLayer("globPool", new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build(), prev)
                .addLayer("output", new OutputLayer.Builder()
                        .nOut(10)
                        .lossFunction(new LossMCXENT())
                        .activation(new ActivationSoftmax()).build(), "globPool");
        return "output";
    }
}
