package examples.cifar10;

import org.deeplearning4j.nn.conf.graph.ScaleVertex;

import static examples.cifar10.LayerUtil.*;

/**
 * An Inception ResNet block C. See figure 13 in https://arxiv.org/pdf/1602.07261.pdf
 *
 * @author Christian Skarby
 */
public class InceptionResNetCBlock implements Block {

    private final String name;
    private final long nrofKernels;

    public InceptionResNetCBlock(String name, long nrofKernels) {
        this.name = name;
        this.nrofKernels = nrofKernels;
    }

    @Override
    public String add(GraphBuilderWrapper builder, String... prev) {

        final String branch0 = new ConvBnRelu(name + ".branch0.1x1",conv1x1(nrofKernels)).add(builder, prev);

        String next = new ConvBnRelu( name + ".branch1.1x1",conv1x1(nrofKernels)).add(builder, prev);
        next = new ConvBnRelu(name + ".branch1.1x3",conv1x3Same(nrofKernels)).add(builder, next);
        final String branch1 = new ConvBnRelu(name + ".branch1.3x1", conv3x1Same(nrofKernels)).add(builder, next);

        builder

                .addLayer(name + ".out", conv1x1(1792), branch0, branch1)
                .addVertex(name + ".scale", new ScaleVertex(0.2), name + ".out");

        return name + ".scale";
    }
}
