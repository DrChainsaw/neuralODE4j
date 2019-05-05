package ode.vertex.conf;

import ode.vertex.impl.gradview.parname.ParamNameMapping;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;

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
    public List<Regularization> getRegularizationByParam(String paramName) {
        final Pair<String, String> vertexAndParam = paramNameMapping.reverseMap(paramName);
        final GraphVertex vertex = graph.getVertex(vertexAndParam.getFirst());
        return vertex.getConfig().getRegularizationByParam(vertexAndParam.getSecond());
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
}
