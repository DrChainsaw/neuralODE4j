package examples.cifar10;

import org.deeplearning4j.nn.conf.layers.Layer;

import static examples.cifar10.LayerUtil.norm;

/**
 * Convolution plus batch norm used by Inception blocks
 *
 * @author Christian Skarby
 */
public class ConvBnRelu implements Block {

    private final String namePrefix;
    private final Layer conv;

    public ConvBnRelu(String namePrefix, Layer conv) {
        this.namePrefix = namePrefix;
        this.conv = conv;
    }

    @Override
    public String add(GraphBuilderWrapper builder, String... prev) {
        builder
                .addLayer(namePrefix + ".conv", conv, prev)
                .addLayer(namePrefix + ".bnRelu",norm(), namePrefix + ".conv");

        return namePrefix + ".bnRelu";
    }

}
