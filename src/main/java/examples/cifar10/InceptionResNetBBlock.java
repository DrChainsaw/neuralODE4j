package examples.cifar10;

import org.deeplearning4j.nn.conf.graph.ScaleVertex;

import static examples.cifar10.LayerUtil.*;

/**
 * An Inception ResNet block A. See figure 11 in https://arxiv.org/pdf/1602.07261.pdf
 *
 * @author Christian Skarby
 */
public class InceptionResNetBBlock implements Block {

    private final String name;
    private final long nrofKernels;

    public InceptionResNetBBlock(String name, long nrofKernels) {
        this.name = name;
        this.nrofKernels = nrofKernels;
    }

    @Override
    public String add(GraphBuilderWrapper builder, String... prev) {
        builder
                .addLayer(name + ".branch0.1x1", conv1x1(nrofKernels), prev)

                .addLayer(name + ".branch1.1x1", conv1x1(nrofKernels), prev)
                .addLayer(name + ".branch1.1x7", conv1x7Same(nrofKernels), name + ".branch1.1x1")
                .addLayer(name + ".branch1.7x1", conv7x1Same(nrofKernels), name + ".branch1.1x7")

                .addLayer(name + ".out", conv1x1(896), name + ".branch0.1x1", name + ".branch1.7x1")
                .addVertex(name + ".scale", new ScaleVertex(0.3), name + ".out");

        return name + ".scale";
    }
}
