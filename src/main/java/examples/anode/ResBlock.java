package examples.anode;

import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;

/**
 * Generic residual block
 *
 * @author Christian Skarby
 */
public class ResBlock implements Block {

    private final String name;
    private final Block blockToRes;

    public ResBlock(String name, Block blockToRes) {
        this.name = name;
        this.blockToRes = blockToRes;
    }

    @Override
    public String add(GraphBuilderWrapper builder, String... prev) {
        final String residual = blockToRes.add(builder, prev);
        builder.addVertex(name + ".add", new ElementWiseVertex(ElementWiseVertex.Op.Add), prev[0], residual);
        return name + ".add";
    }
}
