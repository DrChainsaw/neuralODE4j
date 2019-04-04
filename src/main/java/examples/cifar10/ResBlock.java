package examples.cifar10;

import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.nd4j.linalg.activations.impl.ActivationReLU;

/**
 * Generic residual block
 *
 * @author Christian Skarby
 */
public class ResBlock implements Block {

    private final String name;
    private final Block blockToRes;
    private final boolean addRelu;

    public ResBlock(String name, Block blockToRes) {
        this(name, blockToRes, true);
    }

    public ResBlock(String name, Block blockToRes, boolean addRelu) {
        this.name = name;
        this.blockToRes = blockToRes;
        this.addRelu = addRelu;
    }

    @Override
    public String add(GraphBuilderWrapper builder, String... prev) {
        final String residual = blockToRes.add(builder, prev);

        builder.addVertex(name + ".add", new ElementWiseVertex(ElementWiseVertex.Op.Add), prev[0], residual);
        if(addRelu) {
            builder.addLayer(name + ".relu", new ActivationLayer.Builder().activation(new ActivationReLU()).build(), name + ".add");
            return name + ".relu";
        }
        return name + ".add";
    }
}
