package examples.cifar10;

import org.deeplearning4j.nn.conf.graph.MergeVertex;

import static examples.cifar10.LayerUtil.*;

/**
 * Reduction block for Inception ResNet V1. See figure 12 in https://arxiv.org/pdf/1602.07261.pdf
 */
public class InceptionReductionBBlock implements Block {

    private final String name;

    public InceptionReductionBBlock(String name) {
        this.name = name;
    }

    @Override
    public String add(GraphBuilderWrapper builder, String... prev) {
        builder
                .addLayer(name + ".branch0.mp3x3", maxPool3x3(2,2), prev)

                .addLayer(name + ".branch1.1x1", conv1x1(256), prev)
                .addLayer(name + ".branch1.3x3", conv3x3(384, 2,2),name + ".branch1.1x1")

                .addLayer(name + ".branch2.1x1", conv1x1(256), prev)
                .addLayer(name + ".branch2.3x3", conv3x3(256, 2,2), name + ".branch2.1x1")

                .addLayer(name + ".branch3.1x1", conv1x1(256), prev)
                .addLayer(name + ".branch3.3x3a", conv3x3Same(256), name + ".branch3.1x1")
                .addLayer(name + ".branch3.3x3b", conv3x3(256,2,2), name + ".branch3.3x3a")

                .addVertex(name + ".merge", new MergeVertex(),
                        name + ".branch0.mp3x3",
                        name + ".branch1.3x3",
                        name + ".branch2.3x3",
                        name + ".branch3.3x3b");

        return name + ".merge";
    }
}
