package ode.vertex.conf;

import ode.vertex.impl.gradview.parname.ParamNameMapping;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.primitives.Pair;

/**
 * Basic {@link TrainingConfig} for vertices which do not have a {@link org.deeplearning4j.nn.conf.layers.Layer}
 *
 * @author Christian Skarby
 */
public class DefaultTrainingConfig implements TrainingConfig {

    private final String name;
    private final ComputationGraph graph;
    private final ParamNameMapping paramNameMapping;

    public DefaultTrainingConfig(ComputationGraph graph, String name, ParamNameMapping paramNameMapping) {
        this.name = name;
        this.graph = graph;
        this.paramNameMapping = paramNameMapping;
    }

    @Override
    public String getLayerName() {
        return name;
    }

    @Override
    public boolean isPretrain() {
        return false;
    }

    @Override
    public double getL1ByParam(String paramName) {
        final Pair<String, String> vertexAndParam = paramNameMapping.reverseMap(paramName);
        final GraphVertex vertex = graph.getVertex(vertexAndParam.getFirst());
        return vertex.getConfig().getL1ByParam(vertexAndParam.getSecond());
    }

    @Override
    public double getL2ByParam(String paramName) {
        final Pair<String, String> vertexAndParam = paramNameMapping.reverseMap(paramName);
        final GraphVertex vertex = graph.getVertex(vertexAndParam.getFirst());
        return vertex.getConfig().getL2ByParam(vertexAndParam.getSecond());
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        final Pair<String, String> vertexAndParam = paramNameMapping.reverseMap(paramName);
        final GraphVertex vertex = graph.getVertex(vertexAndParam.getFirst());
        return vertex.getConfig().isPretrainParam(vertexAndParam.getSecond());
    }

    @Override
    public IUpdater getUpdaterByParam(String paramName) {
        final Pair<String, String> vertexAndParam = paramNameMapping.reverseMap(paramName);
        final GraphVertex vertex = graph.getVertex(vertexAndParam.getFirst());
        return vertex.getConfig().getUpdaterByParam(vertexAndParam.getSecond());
    }

    @Override
    public GradientNormalization getGradientNormalization() {
        return GradientNormalization.None;
    }

    @Override
    public double getGradientNormalizationThreshold() {
        return 0;
    }

    @Override
    public void setPretrain(boolean pretrain) {

    }
}
