package examples.cifar10;

import org.deeplearning4j.nn.conf.graph.MergeVertex;

import static examples.cifar10.LayerUtil.*;

/**
 * Reduction block for Inception ResNet V1. See figure 7 in https://arxiv.org/pdf/1602.07261.pdf
 */
public class InceptionReductionABlock implements Block {

    private final String name;

    public InceptionReductionABlock(String name) {
        this.name = name;
    }

    @Override
    public String add(GraphBuilderWrapper builder, String... prev) {
        builder.addLayer(name + ".branch0.mp3x3", maxPool3x3(2,2), prev);

        final String branch1 = new ConvBnRelu(name + ".branch1.3x3", conv3x3(384, 2,2)).add(builder, prev);

        String next = new ConvBnRelu(name + ".branch2.1x1", conv1x1(192)).add(builder, prev);
        next = new ConvBnRelu(name + ".branch2.3x3a", conv3x3Same(192)).add(builder, next);
        final String branch2 =        new ConvBnRelu(name + ".branch2.3x3b", conv3x3(256,2,2)).add(builder, next);

        builder.addVertex(name + ".merge", new MergeVertex(), name + ".branch0.mp3x3", branch1, branch2);

        return name + ".merge";
    }
}
