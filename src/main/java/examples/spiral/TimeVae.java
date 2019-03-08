package examples.spiral;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * A variational auto encoder which takes time as input. Implemented as multiple {@link ComputationGraph}s; one encoder,
 * one latentTime and one decoder. These three share weights with the input {@link ComputationGraph} which has all three
 * parts connected for training purposes
 *
 * @author Christian Skarby
 */
public class TimeVae {

    private final ComputationGraph encoder;
    private final GraphVertex latentTime;
    private final ComputationGraph decoder;

    public TimeVae(ComputationGraph model, String z0, String zt) {
        encoder = createEncoder(model, z0);
        latentTime = createLatentTime(model, zt, encoder.numParams());
        decoder = createDecoder(model, zt, encoder.numParams() + latentTime.numParams());
    }

    private ComputationGraph createEncoder(ComputationGraph model, String z0) {
        final ComputationGraphConfiguration conf = model.getConfiguration();

        final ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder(model.conf()).graphBuilder();
        long nrofParams = 0;
        for (String vertexName : conf.getTopologicalOrderStr()) {
            if (!conf.getNetworkInputs().contains(vertexName) && !conf.getNetworkOutputs().contains(vertexName)) {
                builder.addVertex(vertexName, conf.getVertices().get(vertexName), conf.getVertexInputs().get(vertexName).toArray(new String[0]));
                nrofParams += model.getVertex(vertexName).numParams();
            }
            if (vertexName.equals(z0)) {
                builder.setOutputs(z0);
                builder.addInputs(conf.getNetworkInputs().get(0));
                break;
            }
        }

        final ComputationGraph encoder = new ComputationGraph(builder.build());
        encoder.init(model.params().get(NDArrayIndex.interval(0, nrofParams)), false);

        return encoder;
    }

    private GraphVertex createLatentTime(ComputationGraph model, String zt, long paramStart) {
        final ComputationGraphConfiguration conf = model.getConfiguration();

        final ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder(model.conf()).graphBuilder();

        builder.addVertex(zt, conf.getVertices().get(zt), conf.getVertexInputs().get(zt).toArray(new String[0]));
        builder.setOutputs(zt);
        builder.addInputs("z0", conf.getNetworkInputs().get(1));
        final long nrofParams = model.getVertex(zt).numParams();

        final ComputationGraph latentTime = new ComputationGraph(builder.build());
        latentTime.init(model.params().get(NDArrayIndex.interval(paramStart, paramStart + nrofParams)), false);

        return latentTime.getVertex(zt);
    }

    private ComputationGraph createDecoder(ComputationGraph model, String zt, long paramStart) {
        final ComputationGraphConfiguration conf = model.getConfiguration();

        final ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder(model.conf()).graphBuilder();

        long nrofParams = 0;
        boolean include = false;
        String last = "";
        for (String vertexName : conf.getTopologicalOrderStr()) {

            if (include && !conf.getNetworkOutputs().contains(vertexName)) {
                builder.addVertex(vertexName, conf.getVertices().get(vertexName), conf.getVertexInputs().get(vertexName).toArray(new String[0]));
                nrofParams += model.getVertex(vertexName).numParams();
                last = vertexName;
            }
            include |= vertexName.equals(zt);
        }
        builder.setOutputs(last);
        builder.addInputs(zt);

        final ComputationGraph decoder = new ComputationGraph(builder.build());
        decoder.init(model.params().get(NDArrayIndex.interval(paramStart, paramStart + nrofParams)), false);

        return decoder;
    }

    INDArray encode(INDArray... inputs) {
        return encoder.outputSingle(inputs);
    }

    INDArray timeDependency(INDArray z0, INDArray time) {
        latentTime.setInputs(z0, time);
        return latentTime.doForward(true, LayerWorkspaceMgr.noWorkspacesImmutable());
    }

    INDArray decode(INDArray... inputs) {
        return decoder.outputSingle(inputs);
    }
}
