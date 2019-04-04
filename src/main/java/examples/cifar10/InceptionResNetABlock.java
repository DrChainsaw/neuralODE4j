package examples.cifar10;

import org.deeplearning4j.nn.conf.graph.ScaleVertex;

import static examples.cifar10.LayerUtil.conv1x1;
import static examples.cifar10.LayerUtil.conv3x3Same;

/**
 * An Inception ResNet block A. See figure 10 in https://arxiv.org/pdf/1602.07261.pdf
 *
 * @author Christian Skarby
 */
public class InceptionResNetABlock implements Block {

    private final String name;
    private final long nrofKernels;


    public InceptionResNetABlock(String name, long nrofKernels) {
        this.name = name;
        this.nrofKernels = nrofKernels;
    }


    @Override
    public String add(GraphBuilderWrapper builder, String... prev) {

        final String branch0 = new ConvBnRelu(name + ".branch0.1x1", conv1x1(nrofKernels)).add(builder, prev);

        String next = new ConvBnRelu(name + ".branch1.1x1", conv1x1(nrofKernels)).add(builder, prev);
        final String branch1 = new ConvBnRelu(name + ".branch1.3x3", conv3x3Same(nrofKernels)).add(builder, next);

        next = new ConvBnRelu(name + ".branch2.1x1", conv1x1(nrofKernels)).add(builder, prev);
        next = new ConvBnRelu(name + ".branch2.3x3a", conv3x3Same(nrofKernels)).add(builder, next);
        final String branch2 = new ConvBnRelu(name + ".branch2.3x3b", conv3x3Same(nrofKernels)).add(builder, next);

        builder
                .addLayer(name + ".out", conv1x1(256), branch0, branch1, branch2)
                .addVertex(name + ".scale", new ScaleVertex(0.17), name + ".out");

        return name + ".scale";
    }
}
