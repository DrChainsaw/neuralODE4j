package examples.anode;

import ode.vertex.conf.OdeVertex;
import ode.vertex.conf.ShapeMatchVertex;
import ode.vertex.conf.helper.OdeHelper;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;

/**
 * Generic ODE block which merges current time in ODE solver with input
 *
 * @author Christian Skarby
 */
public class OdeBlockWithTime implements Block {

    private final NeuralNetConfiguration.Builder globalConfig;
    private final Block odeFunc;
    private final OdeHelper odeHelper;

    public OdeBlockWithTime(NeuralNetConfiguration.Builder globalConfig, Block odeFunc, OdeHelper odeHelper) {
        this.globalConfig = globalConfig;
        this.odeFunc = odeFunc;
        this.odeHelper = odeHelper;
    }

    @Override
    public String add(GraphBuilderWrapper builder, String... prev) {

        final OdeVertex.Builder odeBuilder = new OdeVertex.Builder(
                globalConfig,
                "timeConc",
                new ShapeMatchVertex(new MergeVertex()),
                true,
                prev)
                .odeConf(odeHelper);

        odeFunc.add(new GraphBuilderWrapper.Wrap(odeBuilder), "timeConc");
        builder.addVertex("ode", odeBuilder.build(), prev);

        return "ode";
    }
}
