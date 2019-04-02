package examples.cifar10;

import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.nd4j.linalg.activations.impl.ActivationReLU;

/**
 * Generic residual block
 *
 * @author Christian Skarby
 */
public class ResBlock implements Block {

    private final String name;
    private final Block blockToRes;

    public ResBlock(String name, Block blockToRes) {
        this.name = name;
        this.blockToRes = blockToRes;
    }

    @Override
    public String add(GraphBuilderWrapper builder, String... prev) {
        builder.addLayer(name + ".relu", new ActivationLayer.Builder().activation(new ActivationReLU()).build(), prev)
        .addLayer(name + "inNorm", new BatchNormalization.Builder().build(), name + ".relu");
        final String residual = blockToRes.add(builder, name + "inNorm");

        builder.addVertex(name + ".add", new ElementWiseVertex(ElementWiseVertex.Op.Add), name + ".relu", residual);
        builder.addLayer(name+ ".endNorm", new BatchNormalization.Builder().build(), name + ".add");
        return name+ ".endNorm";
    }
}
