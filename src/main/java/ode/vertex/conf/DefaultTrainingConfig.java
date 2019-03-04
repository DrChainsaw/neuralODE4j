package ode.vertex.conf;

import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.learning.config.IUpdater;

/**
 * Basic {@link TrainingConfig} for vertices which do not have a {@link org.deeplearning4j.nn.conf.layers.Layer}
 *
 * @author Christian Skarby
 */
public class DefaultTrainingConfig implements TrainingConfig {

    private final String name;
    private ComputationGraph graph;
    private IUpdater updater;

    public DefaultTrainingConfig(ComputationGraph graph, String name) {
        this.name = name;
        this.graph = graph;
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
        return 0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        return 0;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public IUpdater getUpdaterByParam(String paramName) {
        if(updater == null) {
            for (org.deeplearning4j.nn.graph.vertex.GraphVertex vertex : graph.getVertices()) {
                if (vertex != null && vertex.hasLayer()) {
                    String parname = vertex.paramTable(false).keySet().iterator().next();
                    updater = vertex.getConfig().getUpdaterByParam(parname).clone();
                    break;
                }
            }
        }

        return updater;
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
