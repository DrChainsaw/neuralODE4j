package examples.cifar10;

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

        builder
                .addLayer(name + ".branch0.1x1", conv1x1(nrofKernels), prev)

                .addLayer(name + ".branch1.1x1", conv1x1(nrofKernels), prev)
                .addLayer(name + ".branch1.3x3", conv3x3Same(nrofKernels), name + ".branch1.1x1")

                .addLayer(name + ".branch2.1x1", conv1x1(nrofKernels), prev)
                .addLayer(name + ".branch2.3x3a", conv3x3Same(nrofKernels), name + ".branch2.1x1")
                .addLayer(name + ".branch2.3x3b", conv3x3Same(nrofKernels), name + ".branch2.3x3a")

                .addLayer(name + ".out", conv1x1(256), name + ".branch0.1x1", name + ".branch1.3x3", name + ".branch2.3x3b");

        return name + ".out";
    }
}
