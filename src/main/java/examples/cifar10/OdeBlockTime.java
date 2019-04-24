package examples.cifar10;

import ode.vertex.conf.OdeVertex;
import ode.vertex.conf.ShapeMatchVertex;
import ode.vertex.conf.helper.OdeHelper;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.nd4j.linalg.activations.impl.ActivationReLU;

import java.util.Collections;

/**
 * Generic ODE block which merges current time in ODE solver with input
 *
 * @author Christian Skarby
 */
public class OdeBlockTime implements Block {

    private final NeuralNetConfiguration.Builder globalConfig;
    private final OdeHelper odeHelper;
    private final Block block;
    private final String name;

    public OdeBlockTime(NeuralNetConfiguration.Builder globalConfig, OdeHelper odeHelper, Block block, String name) {
        this.globalConfig = globalConfig;
        this.odeHelper = odeHelper;
        this.block = block;
        this.name = name;
    }

    @Override
    public String add(GraphBuilderWrapper builder, String... prev) {

        final OdeVertex.Builder odeBuilder = new OdeVertex.Builder(
                globalConfig,
                "timeConc",
                new ShapeMatchVertex(new MergeVertex(), Collections.singletonMap(1,10L)),
                true,
                prev)
                .odeConf(odeHelper);

        String next = block.add(new GraphBuilderWrapper.Wrap(odeBuilder), "timeConc");
        odeBuilder.addLayer("relu", new ActivationLayer.Builder().activation(new ActivationReLU()).build(), next);

        builder.addVertex(name, odeBuilder.build(), prev);

        return name;
    }
}
