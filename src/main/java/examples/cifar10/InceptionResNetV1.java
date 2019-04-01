package examples.cifar10;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import examples.cifar10.GraphBuilderWrapper.Wrap;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static examples.cifar10.LayerUtil.conv1x1;

/**
 * Implementation of Inception ResNet V1 for CIFAR 10. Does not use the stem as CIFAR images are already 32x32 to begin
 * with.
 * <br>
 * ref: https://arxiv.org/pdf/1602.07261.pdf
 *
 * @author Christian Skarby
 */
@Parameters(commandDescription = "Configuration for image classification using a Inception-ResNet-v1 without the stem")
public class InceptionResNetV1 implements ModelFactory {

    private static final Logger log = LoggerFactory.getLogger(InceptionResNetV1.class);

    @Parameter(names = "-nrofABlocks", description = "Number of type A blocks to insert.")
    private int nrofABlocks = 4;

    @Parameter(names = "-nrofBBlocks", description = "Number of type A blocks to insert.")
    private int nrofBBlocks = 10;

    @Parameter(names = "-nrofCBlocks", description = "Number of type C blocks to insert.")
    private int nrofCBlocks = 5;

    @Override
    public ComputationGraph create() {
        log.info("Create model");


        String next = "input";
        final GraphBuilder builder = LayerUtil.initGraphBuilder(Nd4j.getRandom().nextLong(), next);

        builder.addLayer("resize", conv1x1(256), next);
        next = "resize";

        for (int i = 0; i < nrofABlocks; i++) {
            final String name = "Ablock" + i;
            next = new ResBlock(name, new InceptionResNetABlock(name, 32))
                    .add(new Wrap(builder), next);
        }

        next = new InceptionReductionABlock("reductionA").add(new Wrap(builder), next);

        for (int i = 0; i < nrofBBlocks; i++) {
            final String name = "Bblock" + i;
            next = new ResBlock(name, new InceptionResNetBBlock(name, 128))
                    .add(new Wrap(builder), next);
        }

        next = new InceptionReductionBBlock("reductionB").add(new Wrap(builder), next);

        for (int i = 0; i < nrofCBlocks; i++) {
            final String name = "Cblock" + i;
            next = new ResBlock(name, new InceptionResNetCBlock(name, 128))
                    .add(new Wrap(builder), next);
        }

        next = new Output().add(new GraphBuilderWrapper.Wrap(builder), next);
        builder.setOutputs(next);

        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();
        return graph;
    }

    @Override
    public String name() {
        return
                "inceptionResNetV1"
                        + (nrofABlocks > 0 ? "_A" + nrofABlocks : "")
                        + (nrofBBlocks > 0 ? "_B" + nrofBBlocks : "")
                        + (nrofCBlocks > 0 ? "_C" + nrofCBlocks : "");
    }

    @Override
    public MultiDataSetIterator wrapIter(DataSetIterator iter) {
        return new MultiDataSetIteratorAdapter(iter);
    }
}
