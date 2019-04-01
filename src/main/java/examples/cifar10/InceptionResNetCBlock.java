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
        builder
                .addLayer(name + ".branch0.1x1", conv1x1(nrofKernels), prev)

                .addLayer(name + ".branch1.1x1", conv1x1(nrofKernels), prev)
                .addLayer(name + ".branch1.1x3", conv1x3Same(nrofKernels), name + ".branch1.1x1")
                .addLayer(name + ".branch1.3x1", conv3x1Same(nrofKernels), name + ".branch1.1x3")

                .addLayer(name + ".out", conv1x1(1792), name + ".branch0.1x1", name + ".branch1.3x1")
                .addVertex(name + ".scale", new ScaleVertex(0.1), name + ".out");

        return name + ".scale";
    }
}
